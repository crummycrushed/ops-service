import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore


async def main():
    sample = SingleTurnSample(
        response="The Eiffel Tower in located in India.",
        reference="The Eiffel Tower in located in Paris."
    )

    scorer = BleuScore()
    score = await scorer.single_turn_ascore(sample)
    print("BLEU score: ", score)

asyncio.run(main())