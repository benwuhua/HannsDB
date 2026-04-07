use crate::descriptor::ScalarIndexDescriptor;

#[derive(Debug, Clone, PartialEq)]
pub struct InvertedScalarIndex {
    pub descriptor: ScalarIndexDescriptor,
}

impl InvertedScalarIndex {
    pub fn new(descriptor: ScalarIndexDescriptor) -> Self {
        Self { descriptor }
    }
}
