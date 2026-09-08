from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from lxml import etree

KEYWORDS = (
    "SSI",
    "다수결",
    "N_valid",
    "n_detected",
    "암종",
    "임계값",
    "스파이크",
    "포화",
    "성능",
)

GLOBAL_REPLACEMENTS = {
    "최상위 추정 암종과 암종별 상대 분류 점수": "가장 비슷한 암종 패턴과 암종별 상대 분류값",
    "상대 분류 보조값임을 설명": "",
    "자별 기여 피크는 표시하지 않는다.": "",
}

DOCX_PREFIX_REPLACEMENTS = {
    "업로드된 5개 스펙트럼은": "업로드된 5개 스펙트럼은 신호강도(intensity gate), 스파이크 잡음, 포화 여부 및 replicate 간 상관도(≥ 0.90)를 자동 검사합니다. QC 통과 반복측정 수를 N_valid로 정의하며 N_valid가 3 미만이면 검사를 무효로 처리하고 결과를 표시하지 않습니다. N_valid가 3 이상이더라도 강도 부족, 스파이크 잡음 또는 포화가 하나라도 확인되면 추론을 중단하고 SSI, 내부 기준 초과 반복 수 및 Cancer Type ID를 표시하지 않습니다. 상관계수 미달만 있는 경우에는 N_valid가 3 이상이면 결과 확인을 진행할 수 있으며, 검사 무효 시 동일 검체 재측정 또는 신규 검체 재채취를 권고합니다.",
    "분석 결과 화면에서": "분석 결과 화면에서 추가 확인 권고/기준 미만, 환자 단위 SSI 점수, 유효 반복측정 수(N_valid), 내부 기준 초과 반복 수(n_detected/N_valid) 및 QC 요약을 확인할 수 있습니다. 환자 단위 SSI는 유효 반복측정별 내부 암 관련 신호값의 평균을 0~10으로 변환한 보조 점수입니다. 유효 반복측정 중 내부 모델 확률이 θ=0.60을 초과한 수를 n_detected로 정의하며, n_detected > N_valid/2인 경우 추가 확인 권고로 표시합니다. 예를 들어 N_valid=5이면 n_detected≥3이어야 합니다. 평균 SSI와 최종 결과는 서로 다를 수 있습니다. 내부 모델 평균확률과 θ는 화면·임상 보고서에 표시하지 않습니다. 추가 확인 권고 상태에서만 Cancer Type ID(TOO)의 암종별 상대 분류값을 유효 반복측정 간 평균하고, 성별 제약 후 재정규화하여 가장 비슷한 암종 패턴과 암종별 분류 점수를 보조 정보로 표시합니다. 이 값은 진단 확률이 아닙니다.",
    "내부 고정 분할 성능평가 코호트는": "내부 고정 분할 성능평가 코호트는 총 1,628명[Train 976명(암 719/비암 257), Validation 324명(암 239/비암 85), Test 328명(암 241/비암 87)]입니다. θ=0.60을 적용한 Test Cancer Screening 성능은 민감도 94.61%, 특이도 96.55%, AUROC 0.9925이며, Test 기지 암 환자 241명의 Cancer Type ID(TOO) 정확도는 98.34%, macro-F1은 0.9593입니다. 이 수치는 환자별 반복 스펙트럼을 특징 단계에서 평균 집계한 후향적 내부 고정 분할 추정치이며, 배포 웹앱의 반복별 추론 후 과반수 판정 전체 파이프라인을 직접 평가한 수치가 아닙니다. 암군과 비암군의 모집기관 차이에 따른 병원 교란 가능성이 있어 기관 간 일반화 성능을 입증하지 않습니다. 실제 임상 사용 성능은 독립된 외부 임상 검증으로 확인해야 합니다.",
}

HWPX_COMMON_PREFIX_REPLACEMENTS = {
    "분석 결과 화면에서는": "분석 결과 화면에는 검사 무효·기준 미만·추가 확인 권고 중 하나의 최종 상태와 SSI, 내부 기준 초과 반복 수 및 QC 요약을 표시한다. QC를 통과해 실제 판정에 사용하는 반복측정 횟수를 ‘유효 반복측정 수(N_valid)’라고 한다. 각 유효 반복측정에서 암 관련 신호가 모델의 내부 기준을 넘었는지 확인하고, 그중 기준을 넘은 횟수를 ‘내부 기준 초과 반복 수(n_detected)’라고 한다. 최종 결과는 n_detected가 N_valid의 절반보다 많을 때만 ‘추가 확인 권고’로 표시한다. 예를 들어 유효 반복측정이 5회이면 3회 이상, 4회이면 3회 이상, 3회이면 2회 이상이 내부 기준을 넘어야 한다. SSI는 유효 반복측정에서 계산된 암 관련 신호값을 평균해 0.0~10.0으로 변환한 보조 점수이다. SSI는 신호의 평균 크기를 보여주지만 최종 결과는 내부 기준을 넘은 횟수로 정하므로, SSI 구간과 최종 결과가 서로 다를 수 있다. SSI만으로 최종 결과를 정하지 않는다. 내부 모델 확률(p)과 내부 기준값(θ)은 모델 계산에만 사용하며 화면과 임상 보고서에는 표시하지 않는다. 최종 결과가 ‘추가 확인 권고’일 때만 가장 비슷한 암종 패턴과 암종별 상대 분류 점수를 보조 정보로 표시한다. 이 값은 개인의 암 발생 가능성이나 진단 확률을 의미하지 않는다. 환자별 기여 피크 분석은 현재 기능에 포함되지 않아 표시하지 않는다.",
    "SSI는 낮음": "환자 단위 SSI는 QC 통과 반복측정별 내부 암 관련 신호값의 평균을 0.0~10.0으로 변환한 보조 점수이며, 낮음(0.0≤SSI<1.0), 중간(1.0≤SSI<4.0), 높음(4.0≤SSI≤10.0)의 3개 구간으로 설명한다. 최종 결과는 평균 SSI가 아니라 n_detected > N_valid/2의 과반수 규칙으로 정하므로 두 결과가 다를 수 있다. 구간별 관측 암 비율과 표본 수는 검증 데이터의 참고 결과이며 개인별 확진 확률이 아니다.",
    "추가 확인 권고 상태에서는": "추가 확인 권고 상태에서만 QC 통과 반복측정의 암종별 상대 분류값을 평균하고 성별 제약 후 재정규화하여 가장 비슷한 암종 패턴과 암종별 분류 점수를 보조 정보로 표시한다. 암종별 상대 분류값은 개별 암종의 진단 확률이나 확진 신뢰도를 의미하지 않는다.",
    "PDF 보고서는 환자 정보": "PDF 보고서는 환자 정보, 검사 조건, QC 요약, SSI 및 3개 구간의 의미, 내부 기준 초과 반복 수, 최종 결과와 권고 문구를 제공한다. CSV 내보내기는 환자 기본정보, QC, SSI, 내부 기준 초과 반복 수, 최종 결과와 추가 확인 권고 시 암종별 상대 분류값을 제공한다. 내부 모델 평균확률 p와 내부 임계값 θ는 화면·임상 보고서에 표시하지 않는다.",
    "· SSI —": "· SSI — QC 통과 반복측정별 내부 암 관련 신호값의 평균을 0.0~10.0으로 변환한 환자 단위 보조 점수",
    "· final_decision —": "· final_decision — QC 유효 시 n_detected > N_valid/2의 반복측정 과반수 규칙으로 정한 추가 확인 권고/기준 미만 상태",
    "·  cancer_type_probabilities —": "· cancer_type_probabilities — 추가 확인 권고 시 표시하는 7개 암종별 상대 분류값(유효 반복측정 평균, 성별 제약 후 재정규화)",
    "·  cancer_type_prediction —": "· cancer_type_prediction — 추가 확인 권고 시 상대 분류값 argmax로 정한 가장 비슷한 암종 패턴",
    "· n_detected —": "· n_detected — 내부 확률 기준값 θ=0.60을 초과한 QC 통과 반복측정 수",
    "· majority_vote —": "· majority_vote — n_detected/N_valid 형식의 내부 기준 초과 반복 수이며 최종 결과의 판정 근거",
    "· cancer_type_prediction —": "· cancer_type_prediction — 추가 확인 권고 상태에서만 표시하는 가장 비슷한 암종 패턴 보조 정보",
    "주: 내부 성능평가는": "주: 내부 성능평가는 총 1,628명(암 1,199명, 비암 429명)을 환자 단위로 Train 976명, Validation 324명 및 Test 328명으로 층화 분할하여 수행하였다. Validation set에서 선정한 고정 임계값 θ=0.60을 Test set에 적용한 암/비암 선별 성능은 AUROC 0.9925, 민감도 94.61%(228/241), 특이도 96.55%(84/87)였다. 이 평가는 환자별 반복 스펙트럼을 특징 단계에서 평균한 뒤 추론한 결과로, 배포 웹앱의 반복별 추론 후 과반수 판정 전체 파이프라인을 직접 평가한 수치가 아니다.",
    "Test set 암환자 241명을": "Test set 기지 암 환자 241명의 7종암 Cancer Type ID(TOO) 성능은 정확도 98.34%(237/241), macro-F1 0.9593이었다. 이는 내부 후향적 평가 결과이며 암군과 비암군의 모집기관 차이에 따른 병원 교란 가능성이 있어 기관 간 일반화 성능을 입증하지 않는다. 실제 임상 성능은 별도의 독립 외부·전향적 임상 검증으로 확인해야 한다.",
}

HWPX_FILE_PREFIX_REPLACEMENTS = {
    "사용사양서": {},
    "위해요인": {
        "SSI와 3개 구간을 확인하고": "QC를 통과한 반복측정 수(N_valid)와 그중 내부 기준을 넘은 횟수(n_detected)를 확인하고, 기준을 넘은 횟수가 유효 반복측정 수의 절반보다 많을 때 ‘추가 확인 권고’가 된다고 설명한다. SSI는 유효 반복측정의 평균 신호 크기를 보여주는 별도의 보조 점수이므로 최종 결과와 다를 수 있으며, 암종별 상대 분류 점수는 개인별 암 확률이 아님을 설명한다.",
        "환자 단위 SSI와 내부 기준 초과 반복 수": "QC를 통과한 반복측정 수(N_valid)와 그중 내부 기준을 넘은 횟수(n_detected)를 확인하고, 기준을 넘은 횟수가 유효 반복측정 수의 절반보다 많을 때 ‘추가 확인 권고’가 된다고 설명한다. SSI는 유효 반복측정의 평균 신호 크기를 보여주는 별도의 보조 점수이므로 최종 결과와 다를 수 있으며, 암종별 상대 분류 점수는 개인별 암 확률이 아님을 설명한다.",
    },
    "사용자 인터페이스 사양서": {
        "사용자는 SSI 0.0~10.0": "사용자는 환자 단위 SSI 0.0~10.0 점수와 최종 반복측정 과반수 결과, 추가 확인 권고 시 암종 분류 보조정보를 구분해 이해해야 한다.",
        "사용자는 환자 단위 SSI": "사용자는 SSI가 유효 반복측정의 평균 신호를 보여주는 보조 점수이고, 최종 결과는 내부 기준을 넘은 반복측정 횟수의 과반수로 정해진다는 차이를 이해해야 한다. 또한 암종 분류 정보가 개인별 암 확률이나 확진 결과가 아님을 알아야 한다.",
        "SSI 3개 구간과 관측치의 한계를": "QC를 통과해 실제 판정에 사용하는 반복측정 횟수를 ‘유효 반복측정 수(N_valid)’라고 한다. 각 유효 반복측정에서 암 관련 신호가 모델의 내부 기준을 넘었는지 확인하고, 그중 기준을 넘은 횟수를 ‘내부 기준 초과 반복 수(n_detected)’라고 한다. 최종 결과는 n_detected가 N_valid의 절반보다 많을 때만 ‘추가 확인 권고’로 표시한다. 예를 들어 유효 반복측정이 5회이면 3회 이상, 4회이면 3회 이상, 3회이면 2회 이상이 내부 기준을 넘어야 한다. SSI는 유효 반복측정의 암 관련 신호값을 평균해 0.0~10.0으로 변환한 별도의 보조 점수이다. SSI는 신호의 평균 크기를 보여주고 최종 결과는 내부 기준을 넘은 횟수를 사용하므로, SSI 구간과 최종 결과가 서로 다를 수 있다. SSI만으로 최종 결과를 정하지 않는다. 내부 모델 확률(p)과 내부 기준값(θ)은 모델 계산에만 사용하며 화면과 임상 보고서에는 표시하지 않는다. 최종 결과가 ‘추가 확인 권고’일 때만 가장 비슷한 암종 패턴과 암종별 상대 분류 점수를 보조 정보로 표시한다. 이 값은 개인의 암 발생 가능성이나 진단 확률을 의미하지 않는다. 환자별 기여 피크 분석은 현재 기능에 포함되지 않아 표시하지 않는다.",
        "SSI는 유효 반복측정의 평균 신호 보조 점수이고": "QC를 통과해 실제 판정에 사용하는 반복측정 횟수를 ‘유효 반복측정 수(N_valid)’라고 한다. 각 유효 반복측정에서 암 관련 신호가 모델의 내부 기준을 넘었는지 확인하고, 그중 기준을 넘은 횟수를 ‘내부 기준 초과 반복 수(n_detected)’라고 한다. 최종 결과는 n_detected가 N_valid의 절반보다 많을 때만 ‘추가 확인 권고’로 표시한다. 예를 들어 유효 반복측정이 5회이면 3회 이상, 4회이면 3회 이상, 3회이면 2회 이상이 내부 기준을 넘어야 한다. SSI는 유효 반복측정의 암 관련 신호값을 평균해 0.0~10.0으로 변환한 별도의 보조 점수이다. SSI는 신호의 평균 크기를 보여주고 최종 결과는 내부 기준을 넘은 횟수를 사용하므로, SSI 구간과 최종 결과가 서로 다를 수 있다. SSI만으로 최종 결과를 정하지 않는다. 내부 모델 확률(p)과 내부 기준값(θ)은 모델 계산에만 사용하며 화면과 임상 보고서에는 표시하지 않는다. 최종 결과가 ‘추가 확인 권고’일 때만 가장 비슷한 암종 패턴과 암종별 상대 분류 점수를 보조 정보로 표시한다. 이 값은 개인의 암 발생 가능성이나 진단 확률을 의미하지 않는다. 환자별 기여 피크 분석은 현재 기능에 포함되지 않아 표시하지 않는다.",
    },
    "사용자 인터페이스 평가 계획서": {
        "사용자는 환자 ID, 나이, 성별과": "사용자는 환자 ID, 나이, 성별과 선택 항목인 BMI를 등록한 후 SOLUM Healthcare SERS 소변 스펙트럼 측정 표준절차서에 따라 정량화된 CSV 또는 TXT 스펙트럼 파일을 정확히 5개 업로드한다. 현재 평가 빌드는 신호강도, 스파이크 잡음, 포화와 반복측정 간 상관도를 자동 확인한다. QC 통과 수 N_valid가 3 미만이거나 강도 부족·스파이크 잡음·포화가 하나라도 있으면 검사 무효로 처리하고 SSI·최종 결과·Cancer Type ID를 표시하지 않는다. 상관계수 미달만 있는 경우 N_valid가 3 이상이면 결과 확인을 진행할 수 있다. 검사 무효 시 동일 환자정보가 유지된 세션에서 새 파일 5개를 다시 업로드하며, 동일 검체 재측정 또는 신규 검체 재채취가 권고된다.",
        "검사 무효·기준 미만·추가 확인 권고 상태": "검사 무효·기준 미만·추가 확인 권고 중 하나의 최종 상태와 QC 결과를 확인한다. QC를 통과한 반복측정 수(N_valid)와 그중 내부 기준을 넘은 횟수(n_detected)를 구분하고, n_detected가 N_valid의 절반보다 많을 때만 ‘추가 확인 권고’가 된다고 설명한다. 예를 들어 유효 반복측정이 5회이면 3회 이상, 4회이면 3회 이상, 3회이면 2회 이상이 내부 기준을 넘어야 한다. SSI는 유효 반복측정의 평균 신호 크기를 0.0~10.0으로 나타낸 별도의 보조 점수이므로 최종 결과와 다를 수 있다. 내부 모델 확률과 내부 기준값은 화면·임상 보고서에 표시하지 않으며, 암종별 상대 분류 점수는 개인별 암 확률이나 확진 결과가 아님을 설명한다.",
    },
    "시험 후 설문조사": {
        "SSI 3개 구간과 내부 p·θ": "SSI는 유효 반복측정의 평균 신호 크기를 보여주는 보조 점수이고, 최종 결과는 내부 기준을 넘은 반복측정이 절반보다 많은지로 정한다는 설명은 이해하기 쉬웠습니까?",
        "환자 단위 SSI가 평균 신호 보조 점수이고": "SSI는 유효 반복측정의 평균 신호 크기를 보여주는 보조 점수이고, 최종 결과는 내부 기준을 넘은 반복측정이 절반보다 많은지로 정한다는 설명은 이해하기 쉬웠습니까?",
    },
}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(
        node.text or ""
        for node in paragraph.iter()
        if etree.QName(node).localname == "t"
    ).strip()


def inspect_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        xml_names = [
            name
            for name in archive.namelist()
            if name == "word/document.xml"
            or name.startswith("Contents/section") and name.endswith(".xml")
        ]
        print(f"FILE: {path.name}")
        for xml_name in xml_names:
            root = etree.fromstring(archive.read(xml_name))
            for index, paragraph in enumerate(
                node for node in root.iter() if etree.QName(node).localname == "p"
            ):
                text = paragraph_text(paragraph)
                if text and any(keyword in text for keyword in KEYWORDS):
                    print(f"{xml_name}:{index}: {text}")


def replacement_for(path: Path, text: str) -> str | None:
    if text in GLOBAL_REPLACEMENTS:
        return GLOBAL_REPLACEMENTS[text]
    prefix_maps = [DOCX_PREFIX_REPLACEMENTS] if path.suffix.lower() == ".docx" else [HWPX_COMMON_PREFIX_REPLACEMENTS]
    if path.suffix.lower() == ".hwpx":
        prefix_maps.extend(
            replacements
            for marker, replacements in HWPX_FILE_PREFIX_REPLACEMENTS.items()
            if marker in path.name
        )
    for prefix_map in prefix_maps:
        for prefix, replacement in prefix_map.items():
            if text.startswith(prefix):
                return replacement
    return None


def set_paragraph_text(paragraph: etree._Element, text: str) -> None:
    text_nodes = [
        node for node in paragraph.iter() if etree.QName(node).localname == "t"
    ]
    if not text_nodes:
        return
    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = None


def apply_archive(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        payloads = {info.filename: archive.read(info.filename) for info in infos}

    changed = 0
    for xml_name, payload in list(payloads.items()):
        if not (
            xml_name == "word/document.xml"
            or xml_name.startswith("Contents/section") and xml_name.endswith(".xml")
        ):
            continue
        root = etree.fromstring(payload)
        xml_changed = False
        for paragraph in (
            node for node in root.iter() if etree.QName(node).localname == "p"
        ):
            current = paragraph_text(paragraph)
            replacement = replacement_for(path, current)
            if replacement is None or replacement == current:
                continue
            set_paragraph_text(paragraph, replacement)
            changed += 1
            xml_changed = True
        if xml_changed:
            payloads[xml_name] = etree.tostring(
                root,
                encoding="UTF-8",
                xml_declaration=True,
                standalone=True,
            )

    if not changed:
        return 0
    backup = path.with_name(f"{path.stem}.backup-before-actual-output-sync-20260715{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_name(f"{path.name}.tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for info in infos:
            archive.writestr(info, payloads[info.filename])
    try:
        os.replace(temporary, path)
    except PermissionError:
        shutil.copyfile(temporary, path)
        temporary.unlink()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if args.apply:
            print(f"UPDATED: {path.name}: {apply_archive(path)} paragraphs")
        else:
            inspect_archive(path)


if __name__ == "__main__":
    main()
