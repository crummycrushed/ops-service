import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RougeScore


async def main():
    sample = SingleTurnSample(
        response="The Eiffel Tower in located in India.",
        reference="The Eiffel Tower in located in Paris."
    )

    scorer = RougeScore()
    score = await scorer.single_turn_ascore(sample)
    print("Rogue score: ", score)

asyncio.run(main())