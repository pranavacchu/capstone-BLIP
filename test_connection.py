from video_search_engine import VideoSearchEngine

# Test connection to Pinecone
engine = VideoSearchEngine()
stats = engine.get_index_stats()
print(f"Connected! Current vectors in database: {stats['total_vectors']}")