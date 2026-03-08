from agnostic_agent.runtime.artifacts import ArtifactEmitter, build_event


def test_build_event_shape():
    evt = build_event(
        run_id="run_1",
        kind="run.started",
        producer="test",
        payload={"x": 1},
    )
    assert evt.run_id == "run_1"
    assert evt.kind == "run.started"
    assert evt.payload["x"] == 1
    assert evt.event_id.startswith("evt_")


def test_artifact_emitter_collects_events():
    emitter = ArtifactEmitter()
    emitter.emit(run_id="run_1", kind="run.started", producer="test")
    emitter.emit(run_id="run_1", kind="run.completed", producer="test")
    events = emitter.list_events()
    assert len(events) == 2
    assert events[0].kind == "run.started"
    assert events[1].kind == "run.completed"
