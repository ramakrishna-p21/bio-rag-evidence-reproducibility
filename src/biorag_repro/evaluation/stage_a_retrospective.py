"""Bounded retrospective publication; the original prospective readers remain unchanged."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import read_verified_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import Label, SetLocus, StageAIdentity, StageAJudgment
from biorag_repro.evaluation.stage_a import (
    _index,
    _scope_titles_for,
    _spans,
    check_item,
    stage_a_dependencies,
    stage_a_ref,
)
from biorag_repro.evaluation.stage_a import validate_judgment as validate_prospective
from biorag_repro.evaluation.stage_a_storage import read_judgment as read_prospective
from biorag_repro.evaluation.stage_records import publish_record, read_record, stage_record_ref
from biorag_repro.generation.environment import check_utc
from biorag_repro.readiness.files import loads_strict

PROTOCOL = "researcher_approved_retrospective_transcription_v1"
APPROVED_DECISIONS_SHA256 = "293f494a3f7a9cc14a68be1fbd9f6f800b8764450f9f983ef514e91b252e1b24"
SESSION_SHA256 = "3b5a1754747b954d79e2d3b7ee4140e94b732d32a24cefae2997a48b48d1c0de"
AUTHORIZATION_SHA256 = "375da084a719766c476093d096197884dbbdbeb603887961ba7ee29bd99ea521"
KIND = "stage_a_retrospective"
# A focused resolution needs separate, explicit researcher approval of its exact bytes.
APPROVED_COMPONENT_RESOLUTIONS: frozenset[str] = frozenset(
    {
        "0343ba07b906bea9d3c78d5fe481a4b33ee24bb712047e10163244147ecceb84",
        "0a0e5c4cbc0257438a82e5233bd81ae3aa1b0663531ea6e1fbb256a4857ec38f",
        "0f10af9e7c8f794dc96c6008b971229c6c9335c2da7856b45be8ab227cae68e7",
        "1327670d9151ca1e39ae3fc4f730d74c613a6e70a068ce7b2d2470dd76f9d345",
        "1395abbe730a168df567686b46a859704fef10742ef26fbbb3dbd3ed38cf255a",
        "14377e2ed4d065bbbffd2ae7d6e2b57da7dba7d255884ba3d05f03a865dc8968",
        "1450fffe5a2ba6643b7ce838c09c033703867edb0fa953d226941a4ae1680b4a",
        "14f6ae53149d3dd6b4067b4a7d280db05ab1601de1cce5d7ce7516d5a05de5d2",
        "152a5e69b9007a71303788f6a5e6150c082c3aa16e749f6770b7ad3c311dca7b",
        "16e4044c12bab1c7ee86be5d8689d3e3b1ef3da10ba75ec2f43f4eff093d3c24",
        "19d46ce044c40517ecb60888cdcaae9429a145477abe5b5a32f94cba9202267e",
        "1ea4dadf968c290fb6221c298e4c0b7ae674bc4234189663aea24c73e72afb5d",
        "1ebb7674cada2f14b982beafdebfd9569e382572446898a6905283e09f31edf1",
        "208f4244e3f9ab9c2e409f80025c6e9717d21161bac58121a895d7f0a83985b0",
        "2153af3c3261a95f409e14bd1efe0c5d88a932e066eea18187b3bdd4f791d043",
        "21dea8850f04842894fa45a9c029f072b411e3c353900ebc93ac4a24891ed4e9",
        "2260937dfcf7f68b6ed01925dd3ae56a512f01c2d01dc2f0933258c6e48e1036",
        "24a52e0f9f853a8e0cf0ab8941e604e8c502dfbe9de2b96bc23d1949151cc872",
        "2994623bb58fd2e660702b90ca0fdc8cf46d0062b2a0d8e6dc100631177e4026",
        "2a249ad098b195be806f7dc30db0401b077a5a8d4711281ed5743c51362ebc1a",
        "2c84ce70a25196b76ef0799616e00cef0fca027ec10b22ef53ef93938ac37636",
        "36ae643c3277e845e321ede94999e12dd2d050773af80e919b3de55aa0311649",
        "38f01e1070de933dddad4426780c17eecde4a1ac2150bd360cad2aa9800bd259",
        "3c00efd73ea534f736d83622079131495e0b3ee4dc497e5f17dad2b5a2039db0",
        "3ce14638682dde49703eb0d6a9aff4d63fbe1523c85652373a79f1685aef5e67",
        "3d08def2014d4f747e81dc63bc5567b761702e8c4a43627bbe2d33ede772ddde",
        "3d37eaaff94d9cd006d1c212c28eaa9ea50d33bcb739c0a3a42ce8a7e53bc628",
        "4633cbd5df3ae5e37dd3f9b5ecd65c1f3d01a74932a7f8701a65233b7ba18f18",
        "48ca8c9af8fa144bab9569f7ce6fef264c91083081f6841f47b8006efa35bcfc",
        "4b2db132b21f39c177933c71e5a683a3b406a44a59f39912ffbec309d88a9b6c",
        "5b862c52f026595b5dd6d951ec67a2595d5eed91cba454decb7606bd3d10b07e",
        "60d6b2202bfdcc3b83bca48afb2212c486516c4dbf4a9b2c22cea3f3797edc3e",
        "61050a10b3499fb32f0aabda17cb45a21f958e10469c56570518906208455a2f",
        "676e81474098c5c63af921d6d4e172541e482ab9f0fda2bf1d50c71ca804bdff",
        "678067ebcd16108251ec39969c4c1d7d198c83b3bf7d10f62181d150afcc837b",
        "6d999274e7fae2155b3bb5bf2ac18cdb60e31ec1c074ced0607e014de232e4ac",
        "714380825e389aee97dfacfdef902eeb47eed0ca953c3541268297495f84c3d7",
        "7235b9a9a513e97ab49a6d6512d092c9a42651919cadfaf7b7a8ea5853617bf0",
        "7345307f8cae0c155857b26e82d0b92fae815a6d0948044616ff4b855c414cd8",
        "75b9f37765a8087a0da4452dfca558c5a02cc2fb79f707f7a628aca78ad6530b",
        "7784765fb0a43ca31af15cb67db084bf572635914b2f8e0bfcfc1110cdc27845",
        "77b29badb7b3825d4e35f171647c41a5ecd3fd0c9b38535f9bcea67f1e5da5fd",
        "7d3a7d07000929d0c561e8aa4393f34d7896dd92152e3c494d20f580b4d98233",
        "7d45d85672f3abbcbe56820b1d15d3794b4d534c719cca6f94d85af6190c27eb",
        "7ef76c623415ac8f0a8f7bc991e3d105a5c61e3be1fe239c3e1569ffa86ef5e5",
        "8237f909b92def3cb957d543af40149b1c0708a43d25aaefbd1c1e0deeff500d",
        "8344ca5ef14a603c9d88cf63fe95a024c6befd2c31663cad65339b0d144144d5",
        "843155bf021fb1f1c911962ed3f904e649cda1e216457293d70c628bb6f0adaf",
        "851863a319a5503a1c3adca3dcc38acdc810da08eb45687a03bc96437107dc97",
        "8590379aed4b73dfc1720d8648bc2b5a9aee25ba2d3628954674801dd4ea965d",
        "85c3ec644df7cbc486e7f7e5219e28d283be2e66f31f7a1c2a593536ded4354f",
        "864dcdde44e79cbee6c9434eeb5bb85e776542696a0d6cce5199592deb36cc19",
        "8b29a3814835a1da5e9a2bf46f2bad1af649707102d2a0e2b0aea2b5828857ff",
        "8dffa8e693ed9fd63f6714eb51a2c4ab13a27b9f642f2eb2249c8921681632f7",
        "9117300a826c5c73d865ab09ffa7c2011f219a973e8af8849b887a4174c32772",
        "9437416c99accacc16b59f4947cbb5b3aacb358b43e38d3fa39fcf64d1125a5b",
        "970f5fdcdc51f64628584076a3bf76aa5aa2a7eeed697e7e4d10106141d4e166",
        "97d4c17cc6ac02afde7437ff1ba4d666c4d8304acb98cb46ccc5ade3aecdd75b",
        "97e30503891d592c78bb723cc00f11e710f49c95d7155cc51461659ffbfce214",
        "9d8ad1006ffcb1bb18693873cfde985d515d11783ccdc322c4307625d7a3af4c",
        "9df07cd91c7256fb720d7b89ac4011039cdb00159f72d52fc5959e5835390b27",
        "a20ebceb749c367af94a93aec8b2453e682d577a3bc8be334d090e63d6ada226",
        "a500b894a0e212f126c26f75698ad83dbc0d2a4e412cdb402b29b1b198a08a27",
        "a721967eed9c83847602dcc3ade98db0ea04679f7fb2b9413813624810d81726",
        "aad8cc232282b78c9c3ff4c7ac50ae0c676e65521594751d0e15172b67d1a52e",
        "abee5cd522d3266a2296395eb6dd034584b8d5a870747753cc3beffd16545445",
        "ac94801f8a740d49271c77b9d31b90520984413e0a969e0cd762419666538bb2",
        "acd1f2754fbe3b68c6fa9fc4b8e1e163f98527d7e5bdf733b9d2d16b9dea7954",
        "ad611dbb9cacd82ae129c42e53eee7285b7ec96937dea7dad31a21d04e648101",
        "ae51044ab6b0ffd2aa8c8da356625aef512f01fd6e05ba1617305b0d00634521",
        "b1e82dccc67c729c3809ae2b89c78776a29e739eab0485481b5180d9c403b17d",
        "b20d52d3e83119652818960256bb3ca9d18e7344c96a20ef169cbaa9ce15cc29",
        "b2b6ccf8520c67ae6a27d0aa6929b467022fc5c0854d9d2d9d1cafebe0bcc726",
        "b5f5637e8b139e4c40b8caa9b3206b1b190d2b6a9ae9b85649ca354d07dd993a",
        "b68fb4338077c2dd0602880750d7afcf6e24813f204177a3285e20fe595fa0fc",
        "b6d4d0d0a6d1dbd5068f1f2d9d828ed9ee040181e7fc99efbd5def5756f696c8",
        "b8703a5c94a784a9a179879d757153a1ecaa2c60ec9be73f685dd42538bc09ea",
        "bcb9c4e22a5b3b58548577b009720d44c6c653b91029a5d6419d79090c493597",
        "be4a714ba970f588d6f58e90884060b56e8b161212ccd20b6c30dee8f1156ae8",
        "be915a77bda3dcd3b3910c5214fc8dbc684f7995f193479b8ce5fc6a53697041",
        "c0707de28f15ed30722b7112e84e03ef1170ae4faff94efd033dd25c0a894803",
        "c0f25e7354c185b32ba6b31ea99970461f6e1a44e63b6930723b3ae69e694471",
        "c202fe86d629fb5dd40aac45f122e751fc073b7ea15ca99c11f6c3d73e7405b5",
        "c3096f7077da0e68068b47fbab8b5304439ba13aaa268650b426e0e1fef0adf9",
        "c6da0b8c4b28966f63cc59e8db6862a07e679446b7dbeef3733dfe802399253f",
        "c8cd0e15db328d364124cda5f75b33cf6e66c4a4c3717b3de9627afae2d4e56e",
        "cf580d40f7bc838e7b190f0cac046eadf09cf061a61866c840e9bfcf5eb64fde",
        "d0e88033005da66b5f00e6c222ceb999912d28223ffdf582a2b12125e16a4b2b",
        "d39681bb2c294b403524b053136928df44b5a1c4b3baa349c2ce478f835c96f3",
        "d5a258660a1a31ff296b032da233bd8dbbd26dc956707ea1fd1abd02fd01e4dd",
        "d793d7d4cd79f22467dde346788f662ea5dc81a4024b2be9d1974b772a7c6752",
        "df069f99b98ab8008670bcfaf3e8d56de3ae3dc6ec476e6b89e6b672b8b709c9",
        "dfe7e8342365a551c29eb8eedc3df3c3030e9679924a423399465ffd4a33b9c6",
        "dff30c6a9bccc509561f7b868d59ca688b7f32425e6e0b3fea3ad0f814f8d201",
        "e6466cfa27ca22a48ca0b30f49f7e0fb3ba1cc40f37a66efb6a29bb66b3bb53e",
        "e77fee91b77ed33fc9034f739132336a361222b29a96a61b661409df7646467b",
        "eb3b2c3b1a68945d52c2686ec59b5d61c3b628ead0879b7e98b74b952dd414d3",
        "edfff03fab12a594cac62a7b274c75785943b42146621ee4688e0a1ed3d11b4c",
        "ef7e26207f6bfabf8790335cd43db8d4f5ee765d8180b7e379a35c7168c50f00",
        "f048bab1e2f2381d3632c02632e86f9f5e07f385a52fa58d62ecc5fc85f1cb26",
        "f2221ad3703782a57c1872c8936639e4b2efcefe73fbbfdedf2ed7a503f75d68",
        "fcf51c3c13f5f10fa7474383df784fd3725ac87811ba7c26c7d4b063c0773905",
        "fd4da9d3a32a7a524ec20bd7bf53b02d6faa89e56d21b374d7ce0004571bac54",
    }
)


class RetrospectiveJudgment(StageAIdentity):
    schema_version: Literal["2"] = "2"
    basis: Literal["researcher_approved_retrospective_transcription"] = (
        "researcher_approved_retrospective_transcription"
    )
    protocol: Literal["researcher_approved_retrospective_transcription_v1"] = PROTOCOL
    handle: str = Field(pattern=r"R\d{3}")
    label: Label
    candidate: ArtifactRef
    sets: list[SetLocus]
    alteration: str | None = None
    rationale: str = Field(min_length=1)
    reviewer: Literal["Rama Krishna Pudota"] = "Rama Krishna Pudota"
    recorded_at_utc: str
    historical_review_at_utc: None = None
    started_utc: None = None
    finished_utc: None = None
    representation_guess: None = None
    guess_confidence: None = None
    guessed_utc: None = None
    consulted_original_source: None = None
    historical_metadata_status: Literal["unavailable"] = "unavailable"
    eligible_for_review_timing_or_guess_analysis: Literal[False] = False
    approved_decisions: ArtifactRef
    session_manifest: ArtifactRef
    coordinate_map: ArtifactRef
    authorization: ArtifactRef
    component_resolution: ArtifactRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    _utc = field_validator("recorded_at_utc")(check_utc)

    @model_validator(mode="after")
    def evidence_shape(self) -> Self:
        if len({s.set_index for s in self.sets}) != len(self.sets):
            raise ValueError("duplicate evidence-set locus")
        if self.label in {"preserved", "altered_but_sufficient"} and not self.sets:
            raise ValueError("sufficient judgment requires complete component placements")
        if (
            self.label in {"altered_but_sufficient", "scientifically_corrupted"}
            and not self.alteration
        ):
            raise ValueError("altered or corrupted evidence needs its approved description")
        if self.label == "absent" and self.sets:
            raise ValueError("absent evidence has no placements")
        for s in self.sets:
            for p in s.placements:
                if self.component_resolution is None and any(
                    c.equivalence_note is not None or c.answer_surface is not None
                    for c in p.components
                ):
                    raise ValueError("ambiguous equivalence requires focused researcher resolution")
        return self


Judgment = StageAJudgment | RetrospectiveJudgment


def dependencies(value: RetrospectiveJudgment) -> list[tuple[str, ArtifactRef]]:
    result = stage_a_dependencies(value) + [
        ("stage_a_candidate", value.candidate),
        ("approved_decisions", value.approved_decisions),
        ("session_manifest", value.session_manifest),
        ("coordinate_map", value.coordinate_map),
        ("retrospective_authorization", value.authorization),
    ]
    if value.component_resolution is not None:
        result.append(("focused_component_resolution", value.component_resolution))
    return result


def judgment_ref(value: Judgment) -> ArtifactRef:
    if isinstance(value, StageAJudgment):
        return stage_a_ref(value)
    return stage_record_ref(value, KIND, "stage_a", dependencies(value), PROTOCOL)


def _source(ref: ArtifactRef, root: Path) -> dict:
    return loads_strict(read_verified_bytes(root / ref.logical_ref, ref.sha256))


def validate_retrospective(
    value: RetrospectiveJudgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path,
) -> None:
    value = RetrospectiveJudgment.model_validate(value.model_dump())
    check_item(item, document, canonical)
    if (
        value.approved_decisions.sha256,
        value.session_manifest.sha256,
        value.authorization.sha256,
    ) != (
        APPROVED_DECISIONS_SHA256,
        SESSION_SHA256,
        AUTHORIZATION_SHA256,
    ):
        raise ValueError("retrospective publication is outside the approved exact scope")
    approval = _source(value.approved_decisions, root)
    session = _source(value.session_manifest, root)
    authorization = _source(value.authorization, root)
    if value.coordinate_map.sha256 != session["unmasked_evidence_sha256"]:
        raise ValueError("coordinate map differs from the approved session")
    coordinates = _source(value.coordinate_map, root)
    rows = {r["handle"]: r for r in approval["records"]}
    if len(rows) != 156 or value.handle not in rows or value.handle not in coordinates:
        raise ValueError("judgment does not name one of the 156 approved handles")
    row, coordinate = rows[value.handle], coordinates[value.handle]
    if (
        row["researcher_concurred_label"] != value.label
        or row["researcher_concurred_rationale"] != value.rationale
        or coordinate["item_sha256"] != value.item_sha256
        or coordinate["representation"] != value.representation
        or ArtifactRef(**coordinate["canonical"]) != value.canonical
        or authorization["approved_decisions_sha256"] != value.approved_decisions.sha256
        or authorization["protocol"] != value.protocol
    ):
        raise ValueError("retrospective judgment changes approved science or coordinates")
    expected_alteration = (
        value.rationale
        if value.label in {"altered_but_sufficient", "scientifically_corrupted"}
        else None
    )
    if value.alteration != expected_alteration:
        raise ValueError("alteration description differs from the approved rationale")
    # Reuse scientific validation without constructing prospective review metadata.
    validate_prospective(value, item, document, canonical)
    if value.component_resolution is not None:
        if value.component_resolution.sha256 not in APPROVED_COMPONENT_RESOLUTIONS:
            raise ValueError("focused component resolution has no registered researcher approval")
        resolution = _source(value.component_resolution, root)
        if (
            resolution["protocol"] != PROTOCOL
            or resolution["handle"] != value.handle
            or resolution["approved_decisions_sha256"] != value.approved_decisions.sha256
            or resolution["item_sha256"] != value.item_sha256
            or resolution["canonical_sha256"] != value.canonical.sha256
            or resolution["sets"] != [s.model_dump(mode="json") for s in value.sets]
        ):
            raise ValueError("focused resolution names different evidence or coordinates")
        index = _index(document, canonical)
        for s in value.sets:
            for p in s.placements:
                for c in p.components:
                    matches = [
                        q
                        for q in resolution["quotes"]
                        if (q["set_index"], q["container_node_id"], q["role"], q["component_text"])
                        == (s.set_index, p.container_node_id, c.role, c.text)
                    ]
                    if len(matches) != 1 or not matches[0]["canonical_quote"]:
                        raise ValueError("focused resolution needs one genuine quote per component")
                    quote = matches[0]["canonical_quote"]
                    containers = {p.container_node_id} | _scope_titles_for(
                        index, p.container_node_id, item, s.set_index
                    )
                    anchored = {
                        i
                        for container in containers
                        for unit in index.units[container]
                        for i in unit.ids(_spans(unit.text, quote, bounded=False))
                    }
                    if not set(c.occurrence_ids) <= anchored:
                        raise ValueError(
                            "focused component quote does not carry its occurrence anchors"
                        )


def publish_judgment(
    value: RetrospectiveJudgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_retrospective(value, item, document, canonical, root)
    return publish_record(
        value, judgment_ref(value), KIND, "stage_a", dependencies(value), PROTOCOL, root
    )


def read_judgment(
    ref: ArtifactRef,
    root: Path,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> Judgment:
    if not ref.logical_ref.startswith(f"evaluation/stage_a/{KIND}/"):
        return read_prospective(ref, root, item, document, canonical)
    value = read_record(ref, root, RetrospectiveJudgment, judgment_ref, "retrospective Stage A")
    validate_retrospective(value, item, document, canonical, root)
    return value


def validate_scientific_judgment(
    value: Judgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path | None = None,
) -> None:
    if isinstance(value, RetrospectiveJudgment):
        if root is None:
            raise ValueError("retrospective evidence requires its approval artifact root")
        checked = RetrospectiveJudgment.model_validate(value.model_dump())
        validate_retrospective(checked, item, document, canonical, root)
    else:
        validate_prospective(value, item, document, canonical)
