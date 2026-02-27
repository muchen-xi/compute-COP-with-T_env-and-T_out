# Copyright (c) 2026 晨曦
# This code is licensed under the MIT License. See LICENSE file for details.
import pandas as pd
import numpy as np
from tqdm import tqdm






def extract_data(filepath):
    """读取Excel，重点修复：只把数字当环境温度，文字标签跳过"""
    # 【核心修改1】先读取前4行，看看哪行是环境温度（数字），哪行是标签（文字）
    test_df = pd.read_excel(filepath, header=None, nrows=4)
    # 找环境温度行（包含数字，比如-26、-20）和标签行（包含“热量kW”“功率kW”）
    temp_row = None  # 存环境温度（数字）
    label_row = None  # 存标签（文字）
    for idx, row in test_df.iterrows():
        # 检查这一行是否有环境温度（数字，且包含负数，比如-26）
        has_temp = any(isinstance(val, (int, float)) and val <= 30 and val >= -30 for val in row if pd.notna(val))
        # 检查这一行是否有标签（文字）
        has_label = any(
            isinstance(val, str) and ('热量' in str(val) or '功率' in str(val)) for val in row if pd.notna(val))
        if has_temp and temp_row is None:
            temp_row = row.tolist()
        if has_label and label_row is None:
            label_row = row.tolist()

    # 构建列名：前两列是“机型”“出水温度”，后面是“环境温度+标签”
    col_names = [('机型', ''), ('出水温度', '')]
    current_temp = None
    # 从第3列开始处理（跳过前两列）
    for i in range(2, len(temp_row)):
        # 【核心修改2】只把数字当环境温度，文字/空值跳过
        if pd.notna(temp_row[i]) and isinstance(temp_row[i], (int, float)):
            current_temp = str(int(temp_row[i]))  # 只转数字为环境温度
        # 处理标签（只保留有标签的列）
        if pd.notna(label_row[i]) and isinstance(label_row[i], str):
            col_names.append((current_temp, label_row[i]))

    # 【核心修改3】找数据开始的行（跳过前面的列名/空行，直到找到出水温度为35、40的行）
    data_start_row = 0
    all_rows = pd.read_excel(filepath, header=None)
    for idx, row in all_rows.iterrows():
        # 出水温度列（第2列）有35、40等数字，就是数据开始的行
        if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], (int, float)) and row.iloc[1] in [35, 40, 45, 50, 55]:
            data_start_row = idx
            break

    # 读取数据主体
    df_raw = pd.read_excel(filepath, header=None, skiprows=data_start_row)
    df_raw = df_raw.iloc[:, :len(col_names)]  # 只保留有列名的列
    df_raw.columns = pd.MultiIndex.from_tuples(col_names)

    # 数据清洗
    df_raw[('出水温度', '')] = pd.to_numeric(df_raw[('出水温度', '')], errors='coerce')
    df_raw = df_raw.dropna(subset=[('出水温度', '')])  # 删除空行
    df_raw.set_index(('出水温度', ''), inplace=True)
    df_raw.drop(columns=[('机型', '')], inplace=True)
    # 所有热量/功率列转数值
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # 提取环境温度和出水温度
    env_temps = sorted(set([col[0] for col in df_raw.columns if col[0] not in ['', None]]), key=int)
    outlet_temps = df_raw.index.tolist()

    # 计算y=热量/功率 (即COP)，收集数据
    records = []
    for ot in outlet_temps:
        for et in env_temps:
            heat_col = (et, '热量kW')
            power_col = (et, '功率kW')
            if heat_col in df_raw.columns and power_col in df_raw.columns:
                heat = df_raw.loc[ot, heat_col]
                power = df_raw.loc[ot, power_col]
                if pd.notna(heat) and pd.notna(power) and power != 0:  # 这里防止除以0
                    # =====================================================
                    # 【核心修正】：COP = 热量 / 功率 (之前写反了！)
                    y = heat/power
                    # =====================================================
                    records.append([float(et), float(ot), float(y)])

    return {"data": records}

R_2_1 = 0
R_2_2=0
R_2_ln = 0
R_2_sqrt = 0
R_1_2 = 0
R_1_1 = 0
R_1_ln = 0
R_1_sqrt = 0
True_results_2_1 = []
True_results_2_ln = []
True_results_2_sqrt = []
True_results_1_1 = []
True_results_1_ln = []
True_results_1_sqrt = []
True_results_1_2=[]
True_results_2_2=[]
# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    file = "data.xlsx"



    result = extract_data(file)
    all_data = result["data"]
    result = all_data
    x= 0
    y = 0
    all_cop = []
    for row in result:
        all_cop.append(row[2])
    average_cop = sum(all_cop) / len(all_cop)
    print(average_cop)#至此计算完平均COP，为后续计算R打下基础


    # 用于存储所有的组合

    # 第一层循环：第一个数 a
    # a 最多只能到 38，因为要留给后面 3 个数（39, 40, 41）
    for first in tqdm(range(0,39)):
        # 第二层循环：第二个数 b，必须比 a 大
        for second in range(first + 1, 40):
            # 第三层循环：第三个数 c，必须比 b 大
            for third in range(second + 1, 41):
                # 第四层循环：第四个数 d，必须比 c 大
                for fourth in range(third + 1, 42):
                    for i in range(4):
                        if i == 0:

                            #len_x中，第一位环境温度，第二位出水温度，第三位COP
                            len_1 = result[first]
                            len_2 = result[second]
                            len_3 = result[third]
                            len_4 = result[fourth]
                            left=np.array([
                                   [len_1[0]**2,len_1[0],len_1[1],1],
                                   [len_2[0]**2,len_2[0],len_2[1],1],
                                   [len_3[0]**2,len_3[0],len_3[1],1],
                                   [len_4[0]**2,len_4[0],len_4[1],1]])
                            right = np.array([len_1[2], len_2[2], len_3[2], len_4[2]])

                            try:
                                a,b,c,d = np.linalg.solve(left, right)

                                SSres = 0
                                SStot = 0
                                for compute in result:
                                    true_cop=compute[2]
                                    T_env = compute[0]
                                    T_out = compute[1]
                                    compute_cop = a*(T_env**2)+b*T_env+c*T_out+d
                                    SSres = SSres +(true_cop-compute_cop)**2
                                    SStot = SStot + (true_cop-average_cop)**2
                                if float(1 - SSres/SStot) > float(R_2_1):
                                    R_2_1 = float(1 - SSres/SStot)
                                    True_results_2_1=[a,b,c,d]
                                else:
                                    continue
                            except:
                                pass
                        elif i == 1:
                            # len_x中，第一位环境温度，第二位出水温度，第三位COP
                            len_1 = result[first]
                            len_2 = result[second]
                            len_3 = result[third]
                            len_4 = result[fourth]
                            left = np.array([
                                [len_1[0] ** 2, len_1[0], np.sqrt(len_1[1]), 1],
                                [len_2[0] ** 2, len_2[0], np.sqrt(len_2[1]), 1],
                                [len_3[0] ** 2, len_3[0], np.sqrt(len_3[1]), 1],
                                [len_4[0] ** 2, len_4[0], np.sqrt(len_4[1]), 1]])
                            right = np.array([len_1[2], len_2[2], len_3[2], len_4[2]])

                            try:
                                a, b, c, d = np.linalg.solve(left, right)

                                SSres = 0
                                SStot = 0
                                for compute in result:
                                    true_cop = compute[2]
                                    T_env = compute[0]
                                    T_out = compute[1]
                                    compute_cop = a * (T_env ** 2) + b * T_env + c * np.sqrt(T_out) + d
                                    SSres = SSres + (true_cop - compute_cop) ** 2
                                    SStot = SStot + (true_cop - average_cop) ** 2
                                if float(1 - SSres / SStot) > float(R_2_sqrt):
                                    R_2_sqrt = float(1 - SSres / SStot)
                                    True_results_2_sqrt = [a, b, c, d]
                                else:
                                    continue
                            except:
                                pass
                        elif i == 2:
                            # len_x中，第一位环境温度，第二位出水温度，第三位COP
                            len_1 = result[first]
                            len_2 = result[second]
                            len_3 = result[third]
                            len_4 = result[fourth]
                            left = np.array([
                                [len_1[0] ** 2, len_1[0], np.log(len_1[1]), 1],
                                [len_2[0] ** 2, len_2[0], np.log(len_2[1]), 1],
                                [len_3[0] ** 2, len_3[0], np.log(len_3[1]), 1],
                                [len_4[0] ** 2, len_4[0], np.log(len_4[1]), 1]])
                            right = np.array([len_1[2], len_2[2], len_3[2], len_4[2]])

                            try:
                                a, b, c, d = np.linalg.solve(left, right)

                                SSres = 0
                                SStot = 0
                                for compute in result:
                                    true_cop = compute[2]
                                    T_env = compute[0]
                                    T_out = compute[1]
                                    compute_cop = a * (T_env ** 2) + b * T_env + c *np.log( T_out) + d
                                    SSres = SSres + (true_cop - compute_cop) ** 2
                                    SStot = SStot + (true_cop - average_cop) ** 2
                                if float(1 - SSres / SStot) > float(R_2_ln):
                                    R_2_ln = float(1 - SSres / SStot)
                                    True_results_2_ln = [a, b, c, d]
                                else:
                                    continue
                            except:
                                pass
                        elif i == 3:
                            # len_x中，第一位环境温度，第二位出水温度，第三位COP
                            len_1 = result[first]
                            len_2 = result[second]
                            len_3 = result[third]
                            len_4 = result[fourth]
                            left = np.array([
                                [len_1[0] , len_1[1]**2, len_1[1], 1],
                                [len_2[0] , len_2[1]**2, len_2[1], 1],
                                [len_3[0] , len_3[1]**2, len_3[1], 1],
                                [len_4[0] , len_4[1]**2, len_4[1], 1]])
                            right = np.array([len_1[2], len_2[2], len_3[2], len_4[2]])

                            try:
                                a, b, c, d = np.linalg.solve(left, right)

                                SSres = 0
                                SStot = 0
                                for compute in result:
                                    true_cop = compute[2]
                                    T_env = compute[0]
                                    T_out = compute[1]
                                    compute_cop = a * (T_env) + b *( T_out**2 )+ c * T_out + d
                                    SSres = SSres + (true_cop - compute_cop) ** 2
                                    SStot = SStot + (true_cop - average_cop) ** 2
                                if float(1 - SSres / SStot) > float(R_1_2):
                                    R_1_2 = float(1 - SSres / SStot)
                                    True_results_1_2 = [a, b, c, d]
                                else:
                                    continue
                            except:
                                pass
    print("第一大组计算完成")

    for first in tqdm(range(0, 40)):  # 第一个数 i：最小0，最大39（留位置给 j 和 k）
        for second in range(first + 1, 41):  # 第二个数 j：比 i 大，最小 i+1，最大40
            for third in range(second + 1, 42):  # 第三个数 k：比 j 大，最小 j+1，最大41
                for i in range(3):
                    if i == 0:
                        # len_x中，第一位环境温度，第二位出水温度，第三位COP
                        len_1 = result[first]
                        len_2 = result[second]
                        len_3 = result[third]
                        left = np.array([
                            [len_1[0], len_1[1], 1],
                            [len_2[0], len_2[1], 1],
                            [len_3[0], len_3[1], 1]])
                        right = np.array([len_1[2], len_2[2], len_3[2]])

                        try:
                            a, b, c = np.linalg.solve(left, right)

                            SSres = 0
                            SStot = 0
                            for compute in result:
                                true_cop = compute[2]
                                T_env = compute[0]
                                T_out = compute[1]
                                compute_cop =   a * T_env + b * T_out + c
                                SSres = SSres + (true_cop - compute_cop) ** 2
                                SStot = SStot + (true_cop - average_cop) ** 2
                            if float(1 - SSres / SStot) > float(R_1_1):
                                R_1_1 = float(1 - SSres / SStot)
                                True_results_1_1 = [a, b, c]
                            else:
                                continue
                        except:
                            pass
                    elif i == 1:
                        # len_x中，第一位环境温度，第二位出水温度，第三位COP
                        len_1 = result[first]
                        len_2 = result[second]
                        len_3 = result[third]
                        left = np.array([
                            [len_1[0], np.sqrt(len_1[1]), 1],
                            [len_2[0], np.sqrt(len_2[1]), 1],
                            [len_3[0], np.sqrt(len_3[1]), 1]])
                        right = np.array([len_1[2], len_2[2], len_3[2]])

                        try:
                            a, b, c = np.linalg.solve(left, right)

                            SSres = 0
                            SStot = 0
                            for compute in result:
                                true_cop = compute[2]
                                T_env = compute[0]
                                T_out = compute[1]
                                compute_cop = a * T_env + b * np.sqrt(T_out) + c
                                SSres = SSres + (true_cop - compute_cop) ** 2
                                SStot = SStot + (true_cop - average_cop) ** 2
                            if float(1 - SSres / SStot) > float(R_1_sqrt):
                                R_1_sqrt = float(1 - SSres / SStot)
                                True_results_1_sqrt = [a, b, c]
                            else:
                                continue
                        except:
                            pass
                    elif i == 2:
                        # len_x中，第一位环境温度，第二位出水温度，第三位COP
                        len_1 = result[first]
                        len_2 = result[second]
                        len_3 = result[third]
                        left = np.array([
                            [len_1[0], np.log(len_1[1]), 1],
                            [len_2[0], np.log(len_2[1]), 1],
                            [len_3[0], np.log(len_3[1]), 1]])
                        right = np.array([len_1[2], len_2[2], len_3[2]])

                        try:
                            a, b, c = np.linalg.solve(left, right)

                            SSres = 0
                            SStot = 0
                            for compute in result:
                                true_cop = compute[2]
                                T_env = compute[0]
                                T_out = compute[1]
                                compute_cop = a * T_env + b * np.log(T_out) + c
                                SSres = SSres + (true_cop - compute_cop) ** 2
                                SStot = SStot + (true_cop - average_cop) ** 2
                            if float(1 - SSres / SStot) > float(R_1_ln):
                                R_1_ln = float(1 - SSres / SStot)
                                True_results_1_ln = [a, b, c]
                            else:
                                continue
                        except:
                            pass
    print("第二组大推演完成，准备第三组双非线性推导")
    # 五层循环：通过"后数 > 前数"保证组合唯一且无序
    for first in tqdm(range(0, 38)):  # 第1个数：最大只能到37（留4个数给后面）
        for second in range(first + 1, 39):  # 第2个数：从a+1开始，最大到38
            for third in range(second + 1, 40):  # 第3个数：从b+1开始，最大到39
                for fourth in range(third + 1, 41):  # 第4个数：从c+1开始，最大到40
                    for fifth in range(fourth + 1, 42):  # 第5个数：从d+1开始，最大到41
                        len_1 = result[first]
                        len_2 = result[second]
                        len_3 = result[third]
                        len_4 = result[fourth]
                        len_5 = result[fifth]
                        left = np.array([
                            [len_1[0] ** 2, len_1[0], len_1[1]**2,len_1[1],1],
                            [len_2[0] ** 2, len_2[0], len_2[1]**2,len_2[1],1],
                            [len_3[0] ** 2, len_3[0], len_3[1]**2,len_3[1],1],
                            [len_4[0] ** 2, len_4[0], len_4[1]**2,len_4[1],1],
                            [len_5[0] ** 2, len_5[0], len_5[1]**2,len_5[1],1]
                        ])
                        right = np.array([len_1[2], len_2[2], len_3[2], len_4[2],len_5[2]])

                        try:
                            a, b, c, d,e = np.linalg.solve(left, right)

                            SSres = 0
                            SStot = 0
                            for compute in result:
                                true_cop = compute[2]
                                T_env = compute[0]
                                T_out = compute[1]
                                compute_cop = a * (T_env ** 2) + b * T_env + c * (T_out**2 )+ d*T_out+e
                                SSres = SSres + (true_cop - compute_cop) ** 2
                                SStot = SStot + (true_cop - average_cop) ** 2
                            if float(1 - SSres / SStot) > float(R_2_2):
                                R_2_2 = float(1 - SSres / SStot)
                                True_results_2_2 = [a, b, c, d,e]
                            else:
                                continue
                        except:
                            pass
    m = [R_2_1,R_2_2,R_2_ln,R_2_sqrt,R_1_2,R_1_1,R_1_ln,R_1_sqrt]
    result_list = ["R_2_1","R_2_2","R_2_ln","R_2_sqrt","R_1_2","R_1_1","R_1_ln","R_1_sqrt"]
    for i in range(8):
        if m[i]==max(m):
            print(result_list[i]+"R²="+str(m[i]))
    print("R_2_2:R²"+str(R_2_2))
    print("R_1_2:R²" + str(R_1_2))

    # ========== 输出所有模型的最大R² ==========
    print("\n" + "=" * 60)
    print("各模型的最大 R² 值：")
    print("=" * 60)
    print(f"【二次线性】     Q = a·T_env² + b·T_env + c·T_out + d                 → R² = {R_2_1:.6f}")
    print(f"【二次平方根】   Q = a·T_env² + b·T_env + c·√T_out + d                → R² = {R_2_sqrt:.6f}")
    print(f"【二次对数】     Q = a·T_env² + b·T_env + c·ln(T_out) + d             → R² = {R_2_ln:.6f}")
    print(f"【一次+T_out²】  Q = a·T_env + b·T_out² + c·T_out + d                 → R² = {R_1_2:.6f}")
    print(f"【一次线性】     Q = a·T_env + b·T_out + c                            → R² = {R_1_1:.6f}")
    print(f"【一次平方根】   Q = a·T_env + b·√T_out + c                           → R² = {R_1_sqrt:.6f}")
    print(f"【一次对数】     Q = a·T_env + b·ln(T_out) + c                        → R² = {R_1_ln:.6f}")
    print(f"【双二次】       Q = a·T_env² + b·T_env + c·T_out² + d·T_out + e      → R² = {R_2_2:.6f}")

    # ========== 找出最佳模型 ==========
    models = [
        ("二次线性", R_2_1, True_results_2_1, 4),
        ("二次平方根", R_2_sqrt, True_results_2_sqrt, 4),
        ("二次对数", R_2_ln, True_results_2_ln, 4),
        ("一次+T_out²", R_1_2, True_results_1_2, 4),
        ("一次线性", R_1_1, True_results_1_1, 3),
        ("一次平方根", R_1_sqrt, True_results_1_sqrt, 3),
        ("一次对数", R_1_ln, True_results_1_ln, 3),
        ("双二次", R_2_2, True_results_2_2, 5)
    ]

    best = max(models, key=lambda x: x[1])
    name, r2, coeff, n_params = best

    print("\n" + "=" * 60)
    print("🏆 最佳模型")
    print("=" * 60)
    print(f"模型：{name}")
    print(f"R² = {r2:.6f}")

    if n_params == 5:
        a, b, c, d, e = coeff
        print(f"多项式：Q = {a:.6f}·T_env² + {b:.6f}·T_env + {c:.6f}·T_out² + {d:.6f}·T_out + {e:.6f}")
        print(f"系数：a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}, e={e:.6f}")
    elif n_params == 4:
        a, b, c, d = coeff
        print(f"多项式：Q = {a:.6f}·T_env² + {b:.6f}·T_env + {c:.6f}·T_out + {d:.6f}")
        print(f"系数：a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}")
    elif n_params == 3:
        a, b, c = coeff
        print(f"多项式：Q = {a:.6f}·T_env + {b:.6f}·T_out + {c:.6f}")
        print(f"系数：a={a:.6f}, b={b:.6f}, c={c:.6f}")

    print("=" * 60)
