"""
Tegifa Legal — Knowledge Graph Manager
Manages connections to Neo4j for epistemic relationship tracking.
Uses lazy initialization to avoid connection attempts during import/testing.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("tegifa.graph")


class KnowledgeGraphManager:
    """
    Manages connections and transactions to the Neo4j Epistemic Knowledge Graph.
    Uses lazy initialization — the driver is only created on first actual use.
    """

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")
        self._driver = None
        self._initialized = False

    @property
    def driver(self):
        """Lazy-initialize the Neo4j driver on first access."""
        if not self._initialized:
            self._initialized = True
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                self._driver.verify_connectivity()
                logger.info("Connected to Neo4j at %s", self.uri)
            except Exception as e:
                logger.warning("Neo4j unavailable: %s", e)
                self._driver = None
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            self._initialized = False

    def merge_epistemic_relationship(
        self,
        subject: str,
        predicate: str,
        object_node: str,
        confidence: float,
        source: str,
    ) -> str:
        """
        Creates or updates a relational edge between two legal entities.
        E.g., (Case A) -[OVERTURNS {confidence: 0.9}]-> (Statute B)
        """
        if not self.driver:
            return "Graph DB unavailable."

        query = """
        MERGE (s:LegalEntity {name: $subject})
        MERGE (o:LegalEntity {name: $object_node})
        MERGE (s)-[r:RELATION {type: $predicate}]->(o)
        SET r.confidence = $confidence, r.source = $source, r.last_updated = timestamp()
        RETURN type(r) as relationship
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    subject=subject,
                    object_node=object_node,
                    predicate=predicate,
                    confidence=confidence,
                    source=source,
                )
                return f"Merged: {subject} -[{predicate}]-> {object_node}"
        except Exception as e:
            logger.error("Error merging relationship: %s", e)
            return f"Error: {e}"

    def query_relationships(self, entity_name: str) -> str:
        """Retrieve all known relationships for a specific entity."""
        if not self.driver:
            return "Graph DB unavailable."

        query = """
        MATCH (s:LegalEntity {name: $entity_name})-[r:RELATION]->(o:LegalEntity)
        RETURN s.name as subject, r.type as predicate, o.name as object, r.confidence as confidence
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_name=entity_name)
                relationships = [
                    f"({rec['subject']}) -[{rec['predicate']} (Conf: {rec['confidence']})]-> ({rec['object']})"
                    for rec in result
                ]
                if not relationships:
                    return f"No known relationships found for '{entity_name}'."
                return "\n".join(relationships)
        except Exception as e:
            logger.error("Error querying relationships: %s", e)
            return f"Error: {e}"


# Lazy singleton — no connection attempted until first method call
kg_manager = KnowledgeGraphManager()
