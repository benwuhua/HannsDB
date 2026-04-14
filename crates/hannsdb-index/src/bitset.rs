//! Convert per-row pass/fail decisions into a packed `BitsetView` for ANN pre-filtering.
//!
//! This module is intentionally decoupled from `hannsdb-core::query::filter::FilterExpr`
//! to avoid a circular dependency (`hannsdb-core` depends on `hannsdb-index`).  Callers
//! in `hannsdb-core` evaluate their own filter expressions and pass a closure that
//! returns `true` for rows that **pass** the filter.
//!
//! Convention: **bit = 1 means filtered OUT (excluded)**, bit = 0 means kept.
//! This matches the Hanns / BitsetPredicate semantics used by knowhere.

#[cfg(feature = "hanns-backend")]
use hanns::BitsetView;

/// Build a `BitsetView` from a closure that decides whether each row passes a filter.
///
/// * `total`  – number of rows (total bit count).
/// * `passes` – closure called with row index `0..total`; returns `true` if the row
///   **passes** the filter (should be **kept**).
///
/// Returns a `BitsetView` where bit=1 means the row is **excluded**.
#[cfg(feature = "hanns-backend")]
pub fn filter_to_bitset<F>(total: usize, passes: F) -> BitsetView
where
    F: Fn(usize) -> bool,
{
    if total == 0 {
        return BitsetView::new(0);
    }
    let word_count = total.div_ceil(64);
    let mut data = vec![0u64; word_count];
    for i in 0..total {
        if !passes(i) {
            // bit = 1 → excluded
            let word_idx = i / 64;
            let bit_idx = i % 64;
            data[word_idx] |= 1u64 << bit_idx;
        }
    }
    BitsetView::from_vec(data, total)
}

/// Build a packed `Vec<u64>` bitset from a closure (no knowhere dependency).
///
/// Useful when the `hanns-backend` feature is disabled but a compact bitset
/// representation is still needed (e.g. brute-force filtered search).
///
/// Same convention: bit=1 means **excluded**, bit=0 means **kept**.
#[cfg(not(feature = "hanns-backend"))]
pub fn filter_to_bitset_vec<F>(total: usize, passes: F) -> Vec<u64>
where
    F: Fn(usize) -> bool,
{
    if total == 0 {
        return Vec::new();
    }
    let word_count = total.div_ceil(64);
    let mut data = vec![0u64; word_count];
    for i in 0..total {
        if !passes(i) {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            data[word_idx] |= 1u64 << bit_idx;
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hanns-backend")]
    fn collect_bits(bv: &BitsetView) -> Vec<bool> {
        (0..bv.num_bits()).map(|i| bv.get(i)).collect()
    }

    fn bits_from_closure(total: usize, passes: impl Fn(usize) -> bool) -> Vec<bool> {
        #[cfg(feature = "hanns-backend")]
        {
            let bv = filter_to_bitset(total, passes);
            collect_bits(&bv)
        }
        #[cfg(not(feature = "hanns-backend"))]
        {
            let data = filter_to_bitset_vec(total, passes);
            (0..total)
                .map(|i| {
                    let word = data[i / 64];
                    (word >> (i % 64)) & 1 == 1
                })
                .collect()
        }
    }

    #[test]
    fn all_pass_yields_zero_bits() {
        let bits = bits_from_closure(10, |_| true);
        assert!(bits.iter().all(|b| !b));
    }

    #[test]
    fn none_pass_yields_all_bits_set() {
        let bits = bits_from_closure(10, |_| false);
        assert!(bits.iter().all(|b| *b));
    }

    #[test]
    fn selective_exclusion() {
        // 8 rows, exclude indices 1, 3, 7
        let bits = bits_from_closure(8, |i| !matches!(i, 1 | 3 | 7));
        assert!(!bits[0]); // pass
        assert!(bits[1]); // excluded
        assert!(!bits[2]);
        assert!(bits[3]);
        assert!(!bits[4]);
        assert!(!bits[5]);
        assert!(!bits[6]);
        assert!(bits[7]);
    }

    #[test]
    fn spans_multiple_words() {
        // 65 rows – needs two u64 words
        let bits = bits_from_closure(65, |i| i != 64);
        assert!(bits.iter().take(64).all(|b| !b));
        assert!(bits[64]); // row 64 excluded
    }

    #[test]
    fn empty_input() {
        #[cfg(feature = "hanns-backend")]
        {
            let bv = filter_to_bitset(0, |_| false);
            assert_eq!(bv.num_bits(), 0);
        }
        #[cfg(not(feature = "hanns-backend"))]
        {
            let data = filter_to_bitset_vec(0, |_| false);
            assert!(data.is_empty());
        }
    }
}
