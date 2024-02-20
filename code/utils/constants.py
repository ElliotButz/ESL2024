from enum import Enum
    
class Gene(Enum):
    type = [
        'gene ontology',
        'panther',
        'prints',
        'plant ontology',
        'prosite_patterns',
        'prosite_profiles',
        'superfamily',
        'trait ontology',
        'interacts_with',
        'contig',
        'fmin',
        'fmax',
        'strand',
        'annotation score',
        'tmhmm',
        'ncoils',
        'genomic sequence',
        'protein sequence',
        'biotype',
        'description',
        'interpro:description',
        'keyword',
        'trait class',
        'allele',
        'gene name synonyms',
        'family',
        'explanation'
    ]

    features = [
        'contig',
        'fmin',
        'fmax',
        'strand',
        'annotation score',
        'tmhmm',
        'ncoils',
        'genomic sequence',
        'protein sequence',
        'biotype',
        'description',
        'interpro:description',
        'keyword',
        'trait class',
        'allele',
        'gene name synonyms',
        'family',
        'explanation'
    ]

class GO(Enum):
    type = [
        'is_a',
        'namespace',
        'definition',
        'name'
    ]

    features = [
        'namespace',
        'definition',
        'name'
    ]

class PO(Enum):
    type = [
        'is_a',
        'namespace',
        'definition',
        'name'
    ]

    features = [
        'namespace',
        'definition',
        'name'
    ]

class TO(Enum):
    # TO lacks a namespace
    type = [
        'is_a',
        'definition',
        'name'
    ]

    features = [
        'definition',
        'name'
    ]

class PrositeProfiles(Enum):
    type = [
        "prosite_profiles"
    ]

class PrositePatterns(Enum):
    type = [
        "prosite_patterns"
    ]

class SuperFamily(Enum):
    type = [
        "superfamily"
    ]

class Prints(Enum):
    type = [
        "prints"
    ]

class Panther(Enum):
    type = [
        "panther"
    ]

class Node(Enum):
    links = [
        'gene ontology',
        'panther',
        'prints',
        'plant ontology',
        'prosite_patterns',
        'prosite_profiles',
        'superfamily',
        'trait ontology',
        'interacts_with',
        'is_a',
    ]

    features = Gene.features.value + GO.features.value + PO.features.value + TO.features.value