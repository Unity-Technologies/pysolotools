from dataclasses import dataclass

from pysolotools.core.models import Annotation, AnnotationDefinition, DataFactory


@DataFactory.register("type.unity.com/unity.pysolotools.CustomTestType")
@dataclass
class CustomTestType(Annotation):
    int_value: int
    float_value: float
    str_value: str


@DataFactory.register("type.unity.com/unity.pysolotools.CustomTestType")
@dataclass
class CustomTestTypeDefinition(AnnotationDefinition):
    pass


class TestSoloCustomType:
    def test_get_annotation_definitions(self, solo_custom_data_instance):
        defs = solo_custom_data_instance.get_annotation_definitions()
        found = False
        for d in defs.annotationDefinitions:
            if isinstance(d, CustomTestTypeDefinition):
                found = True
                assert d.id == "custom test type"
                assert d.description == "Test type."
        assert found

    def test_frames(self, solo_custom_data_instance):
        found = False
        for f in solo_custom_data_instance.frames():
            caps = f.captures
            assert len(caps) == 1
            for c in caps:
                for a in c.annotations:
                    if isinstance(a, CustomTestType):
                        assert not found
                        found = True
                        assert a.int_value == 42
                        assert a.float_value == 0.17
                        assert a.str_value == "so long"
        assert found
