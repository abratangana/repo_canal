import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient



logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD')
neo4j_database=os.getenv('NEO4J_DATABASE')


from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver

llm_config = LLMConfig(
    api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-4.1-mini",      
    small_model="gpt-4.1-nano", 
    base_url="https://api.openai.com/v1",       
)


async def main():
    
    neo4j_driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database="neo4j"  # Aquí indicamos la base real en Docker
    )

    graphiti = Graphiti(
       graph_driver=neo4j_driver,
       #llm_client=OpenAIGenericClient(config=llm_config),
       #embedder=OpenAIEmbedder(
       #    config=OpenAIEmbedderConfig(
       #       api_key=os.getenv('OPENAI_API_KEY'),
        #       embedding_model="text-embedding-3-small",
        #       base_url="https://api.openai.com/v1",
        #   )
       #),
       #cross_encoder=OpenAIRerankerClient(config=llm_config)
   )


    try:
        await graphiti.build_indices_and_constraints()

        episodes = [
        {
            'content': 'Claude es el asistente de IA insignia de Anthropic. Anteriormente se conocía como Claude Instant en sus versiones anteriores.',
            'type': EpisodeType.text,
            'description': 'Transcripción de pódcast de IA',
        },
    {
        'content': 'Como asistente de IA, Claude ha estado disponible desde el 15 de diciembre de 2022 hasta el presente',
        'type': EpisodeType.text,
        'description': 'Transcripción de pódcast de IA',
    },
    {
        'content': {
            'name': 'GPT-4',
            'creator': 'OpenAI',
            'capability': 'Razonamiento multimodal',
            'previous_version': 'GPT-3.5',
            'training_data_cutoff': 'Abril de 2023',
        },
        'type': EpisodeType.json,
        'description': 'Metadatos del modelo de IA',
    },
    {
        'content': {
            'name': 'GPT-4',
            'release_date': '14 de marzo de 2023',
            'context_window': '128,000 tokens',
            'status': 'Activo',
        },
        'type': EpisodeType.json,
        'description': 'Metadatos del modelo de IA',
    },
]


       
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'AI Agents Unleashed {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content'], ),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: AI Agents Unleashed {i} ({episode["type"].value})')

       

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Which AI assistant is from Anthropic?'")
        results = await graphiti.search('Which AI assistant is from Anthropic?')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Which AI assistant is from Anthropic?', center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='Large Language Models',
            config=node_search_config,
        )

        # Print node search results
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
    
## ARRANQUE EN DOCKER    
#docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data neo4j