from itertools import product
 
# Table 1: Prior P(S)
P_S = {True: 0.30, False: 0.70}
 
# Table 2: P(C | S)
P_C_given_S = {
    True:  {True: 0.05, False: 0.95},   # P(C | S=true)
    False: {True: 0.01, False: 0.99},   # P(C | S=false)
}
 
# Table 3: P(T | C)   ("บวก" = positive, "ลบ" = negative)
P_T_given_C = {
    True:  {"บวก": 0.90, "ลบ": 0.10},   # P(T | C=true)
    False: {"บวก": 0.20, "ลบ": 0.80},   # P(T | C=false)
}
 
 
def joint(s: bool, c: bool, t: str) -> float:
    """P(S=s, C=c, T=t) = P(S=s) * P(C=c | S=s) * P(T=t | C=c)"""
    return P_S[s] * P_C_given_S[s][c] * P_T_given_C[c][t]

JOINT_TABLE = {
    (s, c, t): joint(s, c, t)
    for s, c, t in product([True, False], [True, False], ["บวก", "ลบ"])
}
 
assert abs(sum(JOINT_TABLE.values()) - 1.0) < 1e-9, "ผลรวม joint ต้องเท่ากับ 1"
 
 
def marginal_T(t: str) -> float:
    """P(T=t) = sum over S, C ของ P(S, C, T=t)"""
    return sum(p for (s, c, tt), p in JOINT_TABLE.items() if tt == t)
 
 
def marginal_C_and_T(c: bool, t: str) -> float:
    """P(C=c, T=t) = sum over S ของ P(S, C=c, T=t)"""
    return sum(p for (s, cc, tt), p in JOINT_TABLE.items() if cc == c and tt == t)
 
 
def marginal_S_and_T(s: bool, t: str) -> float:
    """P(S=s, T=t) = sum over C ของ P(S=s, C, T=t)"""
    return sum(p for (ss, c, tt), p in JOINT_TABLE.items() if ss == s and tt == t)
 
 
def posterior_C_given_T(c: bool, t: str) -> float:
    """P(C=c | T=t) = P(C=c, T=t) / P(T=t)"""
    return marginal_C_and_T(c, t) / marginal_T(t)
 
 
def posterior_S_given_T(s: bool, t: str) -> float:
    """P(S=s | T=t) = P(S=s, T=t) / P(T=t)"""
    return marginal_S_and_T(s, t) / marginal_T(t)
 
 
def closest_choice(value: float, choices: dict) -> str:
    """หาตัวเลือก a/b/c/d ที่ใกล้เคียงค่าที่คำนวณได้มากที่สุด"""
    return min(choices, key=lambda k: abs(choices[k] - value))
 
 
if __name__ == "__main__":
    print("=" * 70)
    print("ข้อ 1: P(S=true, C=true, T=\"บวก\")")
    q1 = joint(True, True, "บวก")
    choices_q1 = {"a": 0.135, "b": 0.0135, "c": 0.2154, "d": 0.0205}
    print(f"  = P(S=true) x P(C=true|S=true) x P(T=บวก|C=true)")
    print(f"  = {P_S[True]} x {P_C_given_S[True][True]} x {P_T_given_C[True]['บวก']}")
    print(f"  = {q1:.4f}  ->  เลือก ({closest_choice(q1, choices_q1)}) {choices_q1[closest_choice(q1, choices_q1)]}")
 
    print("\n" + "=" * 70)
    print("ข้อ 42: P(T=\"บวก\")  (รวมทุกกรณีของ S และ C)")
    q42 = marginal_T("บวก")
    choices_q42 = {"a": 0.111, "b": 0.2154, "c": 0.4500, "d": 0.2000}
    for s, c in product([True, False], [True, False]):
        p = joint(s, c, "บวก")
        print(f"  P(S={s}, C={c}, T=บวก) = {p:.4f}")
    print(f"  ผลรวม = {q42:.4f}  ->  เลือก ({closest_choice(q42, choices_q42)}) {choices_q42[closest_choice(q42, choices_q42)]}")
 
    print("\n" + "=" * 70)
    print("ข้อ 43: P(C=true | T=\"บวก\")   (Posterior ตาม Bayes' rule)")
    q43 = posterior_C_given_T(True, "บวก")
    choices_q43 = {"a": 0.032, "b": 0.092, "c": 0.198, "d": 0.500}
    print(f"  = P(C=true, T=บวก) / P(T=บวก)")
    print(f"  = {marginal_C_and_T(True, 'บวก'):.4f} / {q42:.4f}")
    print(f"  = {q43:.4f}  ->  เลือก ({closest_choice(q43, choices_q43)}) {choices_q43[closest_choice(q43, choices_q43)]}")
 
    print("\n" + "=" * 70)
    print("ข้อ 44: P(S=true | T=\"บวก\")   (Posterior ตาม Bayes' rule)")
    q44 = posterior_S_given_T(True, "บวก")
    choices_q44 = {"a": 0.327, "b": 0.092, "c": 0.537, "d": 0.700}
    print(f"  = P(S=true, T=บวก) / P(T=บวก)")
    print(f"  = {marginal_S_and_T(True, 'บวก'):.4f} / {q42:.4f}")
    print(f"  = {q44:.4f}  ->  เลือก ({closest_choice(q44, choices_q44)}) {choices_q44[closest_choice(q44, choices_q44)]}")
 
    print("\n" + "=" * 70)
    print("ข้อ 45: จากผู้ป่วย 1000 คน คาดว่าจะเป็นมะเร็ง (C=true) และผล X-ray บวกกี่ราย?")
    q45_prob = marginal_C_and_T(True, "บวก")
    q45 = q45_prob * 1000
    choices_q45 = {"a": 6, "b": 20, "c": 52, "d": 92}
    print(f"  = P(C=true, T=บวก) x 1000")
    print(f"  = {q45_prob:.4f} x 1000 = {q45:.2f} ราย")
    print(f"  ->  เลือก ({closest_choice(q45, choices_q45)}) \u2248 {choices_q45[closest_choice(q45, choices_q45)]}")
 
    print("\n" + "=" * 70)
    print("สรุปคำตอบทั้งหมด")
    print("=" * 70)
    print(f"  ข้อ 1  : b) 0.0135   (คำนวณได้ {q1:.4f})")
    print(f"  ข้อ 42 : b) 0.2154   (คำนวณได้ {q42:.4f})")
    print(f"  ข้อ 43 : b) 0.092    (คำนวณได้ {q43:.4f})")
    print(f"  ข้อ 44 : a) 0.327    (คำนวณได้ {q44:.4f})")
    print(f"  ข้อ 45 : b) \u2248 20    (คำนวณได้ {q45:.2f})")