from enum import Enum

class IricGene(Enum):
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

class IricGO(Enum):
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

class IricPO(Enum):
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

class IricTO(Enum):
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

class IricPrositeProfiles(Enum):
    type = [
        "prosite_profiles"
    ]

class IricPrositePatterns(Enum):
    type = [
        "prosite_patterns"
    ]

class IricSuperFamily(Enum):
    type = [
        "superfamily"
    ]

class IricPrints(Enum):
    type = [
        "prints"
    ]

class IricPanther(Enum):
    type = [
        "panther"
    ]

class IricNode(Enum):
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

    features = IricGene.features.value + IricGO.features.value + IricPO.features.value + IricTO.features.value

### Arabidopsis thaliana    
class AthalianaGene(Enum):
    type = [
        'interpro',
        'description',
        'end',
        'start',
        'seq_region_name',
        'po',
        'strand',
        'biotype',
        'full_name',
        'curator_summary',
        'go:term',
        'synonym',
        'protein_sequence',
        'genomic_sequence',
    ]

    features = [
        'description',
        'end',
        'start',
        'seq_region_name',
        'strand',
        'biotype',
        'full_name',
        'curator_summary',
        'synonym',
        'protein_sequence',
        'genomic_sequence',
    ]

class AthalianaGO(Enum):
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

class AthalianaPO(Enum):
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

class AthalianaTO(Enum):
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

class AthalianaPrositeProfiles(Enum):
    type = [
        "prosite_profiles"
    ]

class AthalianaPrositePatterns(Enum):
    type = [
        "prosite_patterns"
    ]

class AthalianaSuperFamily(Enum):
    type = [
        "superfamily"
    ]

class AthalianaPrints(Enum):
    type = [
        "prints"
    ]

class AthalianaPanther(Enum):
    type = [
        "panther"
    ]

class AthalianaNode(Enum):
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

    features = AthalianaGene.features.value + AthalianaGO.features.value + AthalianaPO.features.value + AthalianaTO.features.value
