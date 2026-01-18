import re


def are_conflicting_values(val1: str, val2: str) -> bool:
    """
    두 값이 충돌하는지 확인 (주로 숫자 비교)
    """
    # 숫자 추출 (달러, 나이 등)
    def extract_numbers(text: str):
        # $20, 20 dollars, 22살, 35 years old 등에서 숫자 추출
        numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', text)
        return [float(n) for n in numbers]
    
    nums1 = extract_numbers(val1)
    nums2 = extract_numbers(val2)
    
    # 둘 다 숫자가 있고, 다른 경우
    if nums1 and nums2:
        # 주요 숫자가 다르면 충돌
        if nums1[0] != nums2[0]:
            return True
    
    # 텍스트 기반 충돌 (long vs short, sold vs not sold 등)
    conflicting_pairs = [
        ("long", "short"),
        ("sold", "not sold"),
        ("sold", "never sold"),
        ("yes", "no"),
        ("true", "false"),
    ]
    
    val1_lower = val1.lower()
    val2_lower = val2.lower()
    
    for word1, word2 in conflicting_pairs:
        if word1 in val1_lower and word2 in val2_lower:
            return True
        if word2 in val1_lower and word1 in val2_lower:
            return True
    
    return False


def check_internal_conflicts(facts: list, chunk: str) -> dict:
    """
    같은 청크에서 추출된 facts 간 모순 확인
    Returns: conflict dict if found, None otherwise
    """
    if not facts or len(facts) < 2:
        return None
    
    for i, fact1 in enumerate(facts):
        for fact2 in facts[i+1:]:
            # 같은 subject인지 확인 (대소문자 무시, 부분 일치)
            subj1 = str(fact1.get('subject', '')).lower().strip()
            subj2 = str(fact2.get('subject', '')).lower().strip()
            
            # Subject가 비슷하거나 같은 경우
            if subj1 and subj2 and (subj1 == subj2 or subj1 in subj2 or subj2 in subj1):
                # 같은 category인지 확인
                cat1 = fact1.get('category', '')
                cat2 = fact2.get('category', '')
                
                if cat1 == cat2:
                    # effect가 충돌하는지 확인
                    eff1 = str(fact1.get('effect', ''))
                    eff2 = str(fact2.get('effect', ''))
                    
                    if are_conflicting_values(eff1, eff2):
                        return {
                            "is_conflict": True,
                            "conflict_type": "Hard Conflict",
                            "reason": f"Internal contradiction in same chunk: '{eff1}' vs '{eff2}' for subject '{subj1}'",
                            "facts": [fact1, fact2],
                            "conflicting_text": fact2.get('text', chunk),
                            "internal": True  # 내부 충돌 표시
                        }
    
    return None
