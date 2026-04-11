pub mod api;
pub mod routes;
pub mod routes_mutation;
pub mod routes_search;

pub fn startup_banner() -> &'static str {
    "hannsdb-daemon bootstrap"
}

#[cfg(test)]
mod tests {
    #[test]
    fn exposes_startup_banner() {
        assert_eq!(super::startup_banner(), "hannsdb-daemon bootstrap");
    }
}
