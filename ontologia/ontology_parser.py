"""
SEOntology TTL Parser.
Parses seovoc.ttl using rdflib and extracts classes, object properties,
and data properties for mapping to Neo4j schema.
"""

from pathlib import Path
from dataclasses import dataclass, field

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD


SEOVOC = Namespace("https://w3id.org/seovoc/")
SCHEMA = Namespace("http://schema.org/")
DC = Namespace("http://purl.org/dc/elements/1.1/")

# Default TTL path relative to this file
DEFAULT_TTL = Path(__file__).parent / "seontology" / "seovoc.ttl"


@dataclass
class OntologyClass:
    uri: str
    local_name: str
    description: str = ""
    subclass_of: list[str] = field(default_factory=list)


@dataclass
class OntologyProperty:
    uri: str
    local_name: str
    description: str = ""
    domains: list[str] = field(default_factory=list)
    ranges: list[str] = field(default_factory=list)
    is_functional: bool = False
    is_inverse_functional: bool = False
    inverse_of: str | None = None


@dataclass
class ParsedOntology:
    classes: list[OntologyClass]
    object_properties: list[OntologyProperty]
    data_properties: list[OntologyProperty]

    def get_class(self, name: str) -> OntologyClass | None:
        for c in self.classes:
            if c.local_name == name:
                return c
        return None

    def get_data_props_for_class(self, class_name: str) -> list[OntologyProperty]:
        return [
            p for p in self.data_properties
            if class_name in p.domains or f"seovoc:{class_name}" in p.domains
        ]

    def get_object_props_for_class(self, class_name: str) -> list[OntologyProperty]:
        return [
            p for p in self.object_properties
            if class_name in p.domains or f"seovoc:{class_name}" in p.domains
        ]


def _local_name(uri: str) -> str:
    """Extract local name from a URI."""
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def _prefix_name(uri: str) -> str:
    """Return prefixed name like 'seovoc:WebPage' or 'schema:Thing'."""
    s = str(uri)
    if s.startswith(str(SEOVOC)):
        return s.replace(str(SEOVOC), "seovoc:")
    if s.startswith(str(SCHEMA)):
        return s.replace(str(SCHEMA), "schema:")
    if s.startswith(str(OWL)):
        return s.replace(str(OWL), "owl:")
    return _local_name(s)


def _get_description(g: Graph, subject) -> str:
    """Get dc:description for a subject."""
    for _, _, o in g.triples((subject, DC.description, None)):
        val = str(o)
        if not val.startswith("http"):
            return val
    return ""


def _extract_union_members(g: Graph, bnode) -> list[str]:
    """Extract class names from an owl:unionOf blank node."""
    members = []
    from rdflib.collection import Collection
    for _, _, union_list in g.triples((bnode, OWL.unionOf, None)):
        for item in Collection(g, union_list):
            members.append(_local_name(item))
    return members


def parse_seovoc(ttl_path: str | Path | None = None) -> ParsedOntology:
    """
    Parse seovoc.ttl and return structured ontology data.

    Args:
        ttl_path: Path to seovoc.ttl file. Defaults to bundled version.

    Returns:
        ParsedOntology with classes, object properties, and data properties.
    """
    ttl_path = Path(ttl_path) if ttl_path else DEFAULT_TTL
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")

    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    # --- Extract Classes ---
    classes = []
    for s in g.subjects(RDF.type, OWL.Class):
        if isinstance(s, type(next(iter(g.subjects(RDF.type, OWL.Class))))):
            # Skip blank nodes (anonymous classes)
            from rdflib import BNode
            if isinstance(s, BNode):
                continue
        name = _local_name(s)
        desc = _get_description(g, s)
        subclasses = [_local_name(o) for _, _, o in g.triples((s, RDFS.subClassOf, None))]
        classes.append(OntologyClass(
            uri=str(s),
            local_name=name,
            description=desc,
            subclass_of=subclasses,
        ))

    # --- Extract Object Properties ---
    object_props = []
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        from rdflib import BNode
        if isinstance(s, BNode):
            continue
        name = _local_name(s)
        desc = _get_description(g, s)

        # Domains (may be a union)
        domains = []
        for _, _, d in g.triples((s, RDFS.domain, None)):
            if isinstance(d, BNode):
                domains.extend(_extract_union_members(g, d))
            else:
                domains.append(_local_name(d))

        # Ranges
        ranges = []
        for _, _, r in g.triples((s, RDFS.range, None)):
            if isinstance(r, BNode):
                ranges.extend(_extract_union_members(g, r))
            else:
                ranges.append(_local_name(r))

        is_functional = (s, RDF.type, OWL.FunctionalProperty) in g
        is_inv_functional = (s, RDF.type, OWL.InverseFunctionalProperty) in g

        inverse = None
        for _, _, inv in g.triples((s, OWL.inverseOf, None)):
            inverse = _local_name(inv)

        object_props.append(OntologyProperty(
            uri=str(s),
            local_name=name,
            description=desc,
            domains=domains,
            ranges=ranges,
            is_functional=is_functional,
            is_inverse_functional=is_inv_functional,
            inverse_of=inverse,
        ))

    # --- Extract Data Properties ---
    data_props = []
    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        from rdflib import BNode
        if isinstance(s, BNode):
            continue
        name = _local_name(s)
        desc = _get_description(g, s)

        domains = []
        for _, _, d in g.triples((s, RDFS.domain, None)):
            if isinstance(d, BNode):
                domains.extend(_extract_union_members(g, d))
            else:
                domains.append(_local_name(d))

        ranges = []
        for _, _, r in g.triples((s, RDFS.range, None)):
            ranges.append(_local_name(r))

        data_props.append(OntologyProperty(
            uri=str(s),
            local_name=name,
            description=desc,
            domains=domains,
            ranges=ranges,
        ))

    return ParsedOntology(
        classes=classes,
        object_properties=object_props,
        data_properties=data_props,
    )


def print_summary(ontology: ParsedOntology) -> None:
    """Print a human-readable summary of the parsed ontology."""
    print(f"\n{'='*60}")
    print(f"SEOntology (seovoc) Summary")
    print(f"{'='*60}")

    print(f"\nClasses: {len(ontology.classes)}")
    for c in sorted(ontology.classes, key=lambda x: x.local_name):
        parent = f" (subClassOf {', '.join(c.subclass_of)})" if c.subclass_of else ""
        print(f"  - {c.local_name}{parent}")

    print(f"\nObject Properties: {len(ontology.object_properties)}")
    for p in sorted(ontology.object_properties, key=lambda x: x.local_name):
        dom = ", ".join(p.domains) if p.domains else "?"
        rng = ", ".join(p.ranges) if p.ranges else "?"
        flags = []
        if p.is_functional:
            flags.append("F")
        if p.is_inverse_functional:
            flags.append("IF")
        if p.inverse_of:
            flags.append(f"inv={p.inverse_of}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  - {p.local_name}: {dom} -> {rng}{flag_str}")

    print(f"\nData Properties: {len(ontology.data_properties)}")
    for p in sorted(ontology.data_properties, key=lambda x: x.local_name):
        dom = ", ".join(p.domains) if p.domains else "?"
        rng = ", ".join(p.ranges) if p.ranges else "?"
        print(f"  - {p.local_name}: {dom} [{rng}]")


if __name__ == "__main__":
    onto = parse_seovoc()
    print_summary(onto)
