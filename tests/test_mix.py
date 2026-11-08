from __future__ import annotations

from minideepseek.data.mix import BatchSource, WeightedBatchMixer
from minideepseek.model.batch import from_sequences


def _factory(tag: int):
    def factory():
        for index in range(4):
            yield from_sequences([[tag, index, tag + 1]])

    return factory


def test_weighted_mixer_proportions_and_determinism():
    sources = [
        BatchSource("a", _factory(1), weight=0.75),
        BatchSource("b", _factory(2), weight=0.25),
    ]
    first = WeightedBatchMixer(sources, total_batches=100, seed=7)
    counts = {"a": 0, "b": 0}
    batches = list(first)
    assert len(batches) == 100
    for source in sources:
        counts[source.name] = first.schedule.count(sources.index(source))
    assert counts == {"a": 75, "b": 25}
    second = WeightedBatchMixer(
        [BatchSource("a", _factory(1), 0.75), BatchSource("b", _factory(2), 0.25)],
        total_batches=100,
        seed=7,
    )
    assert [batch.input_ids.tolist() for batch in second] == [
        batch.input_ids.tolist() for batch in batches
    ]


def test_weighted_mixer_restarts_exhausted_source():
    source = BatchSource("only", _factory(3), weight=1.0)
    mixer = WeightedBatchMixer([source], total_batches=10, seed=1)
    assert len(list(mixer)) == 10
    assert mixer.restart_counts()["only"] == 2


def test_lockstep_mixer_keeps_each_step_on_one_source():
    sources = [
        BatchSource("a", _factory(1), weight=0.5),
        BatchSource("b", _factory(2), weight=0.5),
    ]
    mixer = WeightedBatchMixer(
        sources, total_batches=40, seed=3, batches_per_step=4, lockstep=True
    )
    assert len(mixer) == 40
    assert len(mixer.schedule) == 10  # one slot per optimizer step
    seen = []
    iterator = iter(mixer)
    for _ in range(40):
        next(iterator)
        seen.append(iterator.current_source)
    for start in range(0, 40, 4):
        assert len(set(seen[start : start + 4])) == 1
    assert mixer.schedule.count(0) == 5
    assert mixer.schedule.count(1) == 5


def test_interleaved_mixer_reports_per_batch_source():
    sources = [
        BatchSource("a", _factory(1), weight=0.5),
        BatchSource("b", _factory(2), weight=0.5),
    ]
    mixer = WeightedBatchMixer(sources, total_batches=8, seed=0)
    iterator = iter(mixer)
    seen = []
    for _ in range(8):
        next(iterator)
        seen.append(iterator.current_source)
    assert set(seen) == {"a", "b"}
    assert mixer.current_index in (0, 1)
