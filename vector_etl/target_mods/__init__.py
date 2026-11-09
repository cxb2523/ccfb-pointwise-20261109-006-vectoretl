def get_target_database(config):
    target_type = config['target_database']
    if target_type == 'Pinecone':
        from .pinecone import PineconeTarget
        return PineconeTarget(config)
    elif target_type == 'Qdrant':
        from .qdrant import QdrantTarget
        return QdrantTarget(config)
    elif target_type == 'Weaviate':
        from .weaviate import WeaviateTarget
        return WeaviateTarget(config)
    elif target_type == 'Single Store':
        from .singlestore import SingleStoreTarget
        return SingleStoreTarget(config)
    elif target_type == 'Supabase':
        from .supabase import SupabaseTarget
        return SupabaseTarget(config)
    elif target_type == 'LanceDB':
        from .lancedb import LanceDBTarget
        return LanceDBTarget(config)
    elif target_type == 'Tembo':
        from .tembo import TemboTarget
        return TemboTarget(config)
    elif target_type == 'MongoDB':
        from .mongodb import MongoDBTarget
        return MongoDBTarget(config)
    elif target_type == 'Neo4j':
        from .neo4j import Neo4jTarget
        return Neo4jTarget(config)
    elif target_type == 'Milvus':
        from .milvus import MilvusTarget
        return MilvusTarget(config)
    else:
        raise ValueError(f"Unsupported target database: {target_type}")
