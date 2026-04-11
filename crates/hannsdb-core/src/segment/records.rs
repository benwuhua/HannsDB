use std::fs::OpenOptions;
use std::io;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use half::f16;

pub fn append_records(path: &Path, dimension: usize, vectors: &[f32]) -> io::Result<usize> {
    if dimension == 0 || vectors.len() % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vectors length must be divisible by dimension and dimension > 0",
        ));
    }

    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    // Write all f32 bytes in one shot instead of 4-byte syscalls.
    // SAFETY: &[f32] → &[u8] via bytemuck-style cast; f32 has no padding.
    let byte_slice =
        unsafe { std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4) };
    writer.write_all(byte_slice)?;
    writer.flush()?;
    Ok(vectors.len() / dimension)
}

pub fn load_records(path: &Path, dimension: usize) -> io::Result<Vec<f32>> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dimension must be > 0",
        ));
    }

    let mut file = OpenOptions::new().read(true).open(path)?;
    let metadata = file.metadata()?;
    let file_len = metadata.len() as usize;

    if file_len % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record file byte length is not aligned to f32",
        ));
    }

    let count = file_len / 4;
    // Allocate aligned buffer and read directly into it.
    let mut buffer: Vec<u8> = Vec::with_capacity(file_len);
    // SAFETY: we read exactly file_len bytes, then interpret as f32.
    unsafe {
        buffer.set_len(file_len);
    }
    file.read_exact(&mut buffer)?;

    if count % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record count is not aligned to dimension",
        ));
    }

    // Zero-copy reinterpret: Vec<u8> → Vec<f32>.
    // SAFETY: alignment of u8 (1) divides alignment of f32 (4), and we verified len % 4 == 0.
    // The Vec<u8> is created from a fresh allocation with capacity >= file_len, so alignment is fine.
    let values: Vec<f32> = unsafe {
        let ptr = buffer.as_mut_ptr() as *mut f32;
        let capacity = buffer.capacity() / 4;
        let len = buffer.len() / 4;
        std::mem::forget(buffer);
        Vec::from_raw_parts(ptr, len, capacity)
    };

    // Convert from native endian bytes to f32 values.
    // On little-endian (x86, ARM) this is a no-op at the hardware level.
    #[cfg(target_endian = "big")]
    {
        for v in &mut values {
            *v = f32::from_le(*v);
        }
    }

    Ok(values)
}

/// Append f32 vectors as f16 (half precision) to a binary file.
/// Each component is stored as 2 bytes (f16), halving storage vs f32.
pub fn append_records_f16(path: &Path, dimension: usize, vectors: &[f32]) -> io::Result<usize> {
    if dimension == 0 || vectors.len() % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vectors length must be divisible by dimension and dimension > 0",
        ));
    }

    let f16_vec: Vec<f16> = vectors.iter().map(|&v| f16::from_f32(v)).collect();
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let byte_slice =
        unsafe { std::slice::from_raw_parts(f16_vec.as_ptr() as *const u8, f16_vec.len() * 2) };
    writer.write_all(byte_slice)?;
    writer.flush()?;
    Ok(vectors.len() / dimension)
}

/// Load f16 vectors from a binary file and convert to f32 for computation.
pub fn load_records_f16(path: &Path, dimension: usize) -> io::Result<Vec<f32>> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dimension must be > 0",
        ));
    }

    let mut file = OpenOptions::new().read(true).open(path)?;
    let metadata = file.metadata()?;
    let file_len = metadata.len() as usize;

    if file_len % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "f16 record file byte length is not aligned to f16",
        ));
    }

    let mut buffer: Vec<u8> = Vec::with_capacity(file_len);
    unsafe {
        buffer.set_len(file_len);
    }
    file.read_exact(&mut buffer)?;

    // Reinterpret as f16 slice.
    if (file_len / 2) % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "f16 record count is not aligned to dimension",
        ));
    }

    let f16_slice =
        unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const f16, file_len / 2) };

    // Convert f16 → f32.
    let values: Vec<f32> = f16_slice.iter().map(|&v| v.to_f32()).collect();
    Ok(values)
}

pub fn append_record_ids(path: &Path, ids: &[i64]) -> io::Result<usize> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let byte_slice =
        unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 8) };
    writer.write_all(byte_slice)?;
    writer.flush()?;
    Ok(ids.len())
}

pub fn load_record_ids(path: &Path) -> io::Result<Vec<i64>> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let metadata = file.metadata()?;
    let file_len = metadata.len() as usize;

    if file_len % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ids file byte length is not aligned to i64",
        ));
    }

    let mut buffer: Vec<u8> = Vec::with_capacity(file_len);
    unsafe {
        buffer.set_len(file_len);
    }
    file.read_exact(&mut buffer)?;

    let ids: Vec<i64> = unsafe {
        let ptr = buffer.as_mut_ptr() as *mut i64;
        let capacity = buffer.capacity() / 8;
        let len = buffer.len() / 8;
        std::mem::forget(buffer);
        Vec::from_raw_parts(ptr, len, capacity)
    };

    #[cfg(target_endian = "big")]
    {
        for v in &mut ids {
            *v = i64::from_le(*v);
        }
    }

    Ok(ids)
}
