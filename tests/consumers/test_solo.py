import pytest

from pysolotools.core.iterators import FramesIterator
from pysolotools.core.models import AnnotationDefinition, DataFactory, DatasetMetadata


class TestSolo:
    def test_get_metadata(self, solo_instance):
        metadata = solo_instance.get_metadata()
        assert isinstance(metadata, DatasetMetadata)

    def test_get_annotation_definitions(self, solo_instance):
        annotation_def_types = list(DataFactory.definition_switcher.values())
        annotation_def_types.append(AnnotationDefinition)
        annotation_definition = solo_instance.get_annotation_definitions()
        for ann_def in annotation_definition.annotationDefinitions:
            assert isinstance(ann_def, tuple(annotation_def_types))

    def test_frames(self, solo_instance):
        frames_iter = solo_instance.frames()
        assert isinstance(frames_iter, FramesIterator)

        with pytest.raises(StopIteration):
            while True:
                next(frames_iter)

    def test_len(self, solo_instance):
        assert len(solo_instance.frames()) == 2

    def test_categories(self, solo_instance):
        cats = solo_instance.categories()
        assert len(cats) == 5
        assert cats[1] == "Crate"
        assert cats[2] == "Cube"
        assert cats[3] == "Box"
        assert cats[4] == "Terrain"
        assert cats[5] == "Character"
