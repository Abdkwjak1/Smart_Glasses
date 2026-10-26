from smart_glasses.registry import PeopleRegistry


def test_empty_registry(tmp_path):
    r = PeopleRegistry(tmp_path / "people.csv")
    assert r.all() == []
    assert r.next_id() == 1
    assert r.get(1) is None


def test_add_assigns_incrementing_ids_and_roundtrips_arabic(tmp_path):
    r = PeopleRegistry(tmp_path / "sub" / "people.csv")
    a, b = r.add("جميل"), r.add("عبد")
    assert (a.id, b.id) == (1, 2)
    assert r.get(2).name == "عبد"
    assert [p.name for p in PeopleRegistry(r.path).all()] == ["جميل", "عبد"]


def test_ids_never_reused_after_gaps(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("Name,ID\nx,1\ny,5\n", encoding="utf-8-sig")
    assert PeopleRegistry(path).next_id() == 6


def test_reads_legacy_data_csv_with_bom():
    from pathlib import Path

    legacy = Path(__file__).parent / "fixtures" / "legacy_people.csv"
    assert [p.id for p in PeopleRegistry(legacy).all()] == [1, 2]
