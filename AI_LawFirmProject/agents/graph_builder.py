import os
import logging
from neo4j import GraphDatabase
import json

logger = logging.getLogger("GraphBuilder")
logger.setLevel(logging.INFO)

class KnowledgeGraphManager:
    """
    Manages connections and transactions to the Neo4j Epistemic Knowledge Graph.
    """
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j Graph Database.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def merge_epistemic_relationship(self, subject: str, predicate: str, object_node: str, confidence: float, source: str):
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
                result = session.run(query, subject=subject, object_node=object_node, predicate=predicate, confidence=confidence, source=source)
                return f"Successfully merged relationship: {subject} -[{predicate}]-> {object_node}"
        except Exception as e:
            logger.error(f"Error merging relationship: {e}")
            return f"Error: {e}"

    def query_relationships(self, entity_name: str):
        """
        Retrieves all known relationships for a specific entity to feed into the LangGraph context.
        """
        if not self.driver:
            return "Graph DB unavailable."

        query = """
        MATCH (s:LegalEntity {name: $entity_name})-[r:RELATION]->(o:LegalEntity)
        RETURN s.name as subject, r.type as predicate, o.name as object, r.confidence as confidence
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_name=entity_name)
                relationships = []
                for record in result:
                    relationships.append(f"({record['subject']}) -[{record['predicate']} (Conf: {record['confidence']})]-> ({record['object']})")

                if not relationships:
                    return f"No known relationships found for '{entity_name}'."
                return "\n".join(relationships)
        except Exception as e:
            logger.error(f"Error querying relationships: {e}")
            return f"Error: {e}"

# Expose a singleton instance
kg_manager = KnowledgeGraphManager()
