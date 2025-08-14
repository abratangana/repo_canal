from neo4j import AsyncGraphDatabase

driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "xxx"))
async def test():
    async with driver.session() as session:
        result = await session.run("RETURN 1 AS test")
        print(await result.single())

import asyncio
asyncio.run(test())
