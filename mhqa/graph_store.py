from neo4j import GraphDatabase
from . import config
from .utils import norm_entity

class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    def close(self): self.driver.close()

    def _run(self, q, **p):
        with self.driver.session() as s:
            return list(s.run(q, **p))

    def reset(self):
        self._run("MATCH (n) DETACH DELETE n")
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self._run("CREATE CONSTRAINT doc_title IF NOT EXISTS FOR (d:Doc) REQUIRE d.title IS UNIQUE")

    def add_doc(self, title: str, text: str, triples: list):
        rows = [{"head": t["head"], "hid": norm_entity(t["head"]), "rel": t["relation"],
                 "tail": t["tail"], "tid": norm_entity(t["tail"])}
                for t in triples if norm_entity(t["head"]) and norm_entity(t["tail"])
                and norm_entity(t["head"]) != norm_entity(t["tail"])]
        self._run("""
            MERGE (d:Doc {title:$title}) SET d.text=$text
            MERGE (te:Entity {id:$tid}) ON CREATE SET te.name=$title
            MERGE (d)-[:MENTIONS]->(te)
            WITH d
            UNWIND $rows AS t
            MERGE (h:Entity {id:t.hid}) ON CREATE SET h.name=t.head
            MERGE (o:Entity {id:t.tid}) ON CREATE SET o.name=t.tail
            MERGE (h)-[:RELATED {type:t.rel, doc:$title}]->(o)
            MERGE (d)-[:MENTIONS]->(h)
            MERGE (d)-[:MENTIONS]->(o)
        """, title=title, text=text, tid=norm_entity(title), rows=rows)

    def find_entities(self, names: list) -> list:
        ids = set()
        for n in names:
            i = norm_entity(n)
            if not i:
                continue
            rec = self._run("MATCH (e:Entity {id:$i}) RETURN e.id AS id", i=i)
            if rec:
                ids.add(rec[0]["id"]); continue
            if len(i) >= 4:  # fuzzy fallback
                import re as _re
                pat = ".*(?<![a-z0-9])" + _re.escape(i) + "(?![a-z0-9]).*"
                rec = self._run("MATCH (e:Entity) WHERE e.id =~ $p RETURN e.id AS id ORDER BY size(e.id) LIMIT 3", p=pat)
                ids.update(r["id"] for r in rec)
        return list(ids)

    def entities_of_docs(self, titles: list, limit: int = 10) -> list:
        rec = self._run("MATCH (d:Doc)-[:MENTIONS]->(e:Entity) WHERE d.title IN $t "
                        "RETURN DISTINCT e.id AS id LIMIT $l", t=titles, l=limit)
        return [r["id"] for r in rec]

    def neighborhood(self, ids: list, hops: int = config.GRAPH_HOPS, limit: int = 200) -> list:
        """Multi-hop traversal: returns edges ordered by distance from the seed entities."""
        hops = max(1, min(int(hops), 3))
        rec = self._run(f"""
            MATCH p=(s:Entity)-[:RELATED*1..{hops}]-(t:Entity)
            WHERE s.id IN $ids
            WITH p ORDER BY length(p) LIMIT $limit
            RETURN [r IN relationships(p) |
                {{head: startNode(r).name, rel: r.type, tail: endNode(r).name, doc: r.doc}}] AS edges
        """, ids=ids, limit=limit)
        seen, out = set(), []
        for r in rec:
            for e in r["edges"]:
                k = (e["head"], e["rel"], e["tail"], e["doc"])
                if k not in seen:
                    seen.add(k); out.append(e)
        return out

    def degrees(self, ids: list) -> dict:
        rec = self._run("UNWIND $ids AS i MATCH (e:Entity {id:i}) "
                        "OPTIONAL MATCH (e)-[r:RELATED]-() RETURN e.id AS id, count(r) AS d", ids=ids)
        return {r["id"]: r["d"] for r in rec}

    def doc_titles(self) -> list:
        return [r["t"] for r in self._run("MATCH (d:Doc) RETURN d.title AS t")]

    def doc_texts(self, titles: list) -> dict:
        rec = self._run("MATCH (d:Doc) WHERE d.title IN $t RETURN d.title AS t, d.text AS x", t=titles)
        return {r["t"]: r["x"] for r in rec}

    def stats(self):
        r = self._run("MATCH (e:Entity) WITH count(e) AS ents "
                      "MATCH ()-[r:RELATED]->() RETURN ents, count(r) AS rels")
        return dict(r[0]) if r else {}
