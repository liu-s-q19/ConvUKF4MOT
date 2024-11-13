from sympy import symbols, diff

# 定义变量
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')

# 定义表达式
A = (q1 + q2 - q1*q2) * (q3 + q4 - q3*q4) * q5

# 计算偏导数
dA_dq1 = diff(A, q1)
dA_dq2 = diff(A, q2)
dA_dq3 = diff(A, q3)
dA_dq4 = diff(A, q4)
dA_dq5 = diff(A, q5)

# 定义变量值
q1_val = 0.04
q2_val = 0.02
q3_val = 0.03
q4_val = 0.4
q5_val = 0.6

# 计算 A 的值
A_val = A.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val, q5: q5_val})

# 计算 qi/A * 对应偏导数
q1_partial = (q1_val / A_val) * dA_dq1.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val, q5: q5_val})
q2_partial = (q2_val / A_val) * dA_dq2.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val, q5: q5_val})
q3_partial = (q3_val / A_val) * dA_dq3.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val, q5: q5_val})
q4_partial = (q4_val / A_val) * dA_dq4.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val, q5: q5_val})
q5_partial = (q5_val / A_val) * dA_dq5.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val, q5: q5_val})

# 输出结果
print("偏导数：")
# 输出偏导数的值
print("偏导数的值：")
print("dA/dq1 =", dA_dq1.evalf(subs={q1: 0.04, q2: 0.02, q3: 0.03, q4: 0.4, q5: 0.6}))
print("dA/dq2 =", dA_dq2.evalf(subs={q1: 0.04, q2: 0.02, q3: 0.03, q4: 0.4, q5: 0.6}))
print("dA/dq3 =", dA_dq3.evalf(subs={q1: 0.04, q2: 0.02, q3: 0.03, q4: 0.4, q5: 0.6}))
print("dA/dq4 =", dA_dq4.evalf(subs={q1: 0.04, q2: 0.02, q3: 0.03, q4: 0.4, q5: 0.6}))
print("dA/dq5 =", dA_dq5.evalf(subs={q1: 0.04, q2: 0.02, q3: 0.03, q4: 0.4, q5: 0.6}))

print("A 的值：", A_val)
print("qi/A * 对应偏导数：")
print("q1/A * dA/dq1 =", q1_partial.evalf())
print("q2/A * dA/dq2 =", q2_partial.evalf())
print("q3/A * dA/dq3 =", q3_partial.evalf())
print("q4/A * dA/dq4 =", q4_partial.evalf())
print("q5/A * dA/dq5 =", q5_partial.evalf())
