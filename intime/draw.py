import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

figsize = (8, 5)
left=0.18
right=0.98
top=0.98
bottom=0.1

def draw_shift():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    dev_ols = np.array([0.25, 0.5, 1, 2, 4])

    params = [etas, dev_ols]
    param_names = ['eta', 'dev_ol']
    locs = [3, 4]

    metrics = ['f1']

    xlabels = [r'Contamination Fraction $\eta$', r'Contamination Deviation $d$']
    methods = ['DTMAD', 'MAD', 'Sn', 'IQR', 'ABP']

    marks = ['-D', '-d', '-v', '-^', '-o']
    colors = ['limegreen', 'blue', 'olive', 'purple', 'orange']

    if not os.path.exists('fig/shift'):
        os.makedirs('fig/shift')

    # error rate
    for i, param_name in enumerate(param_names):
        for metric in metrics:

            dir = f'res/shift/{param_name}/{metric}.csv'
            data = np.loadtxt(dir, delimiter=",", dtype=float)

            plt.figure(figsize=figsize)

            ax = plt.gca()
            spine_lw = 2
            ax.spines['top'].set_linewidth(spine_lw)
            ax.spines['bottom'].set_linewidth(spine_lw)
            ax.spines['left'].set_linewidth(spine_lw)
            ax.spines['right'].set_linewidth(spine_lw)

            for j in range(len(methods)):
                plt.plot(params[i], data[:, j], marks[j], color=colors[j], lw=3, ms=18, label=methods[j])

            # plt.xlabel(xlabels[i], {"size": 32})
            plt.ylabel('F1 Score', {"size": 30})
            plt.xticks(fontsize=24)
            if i == 1:
                plt.yticks(fontsize=24)
            else:
                y_ticks = np.arange(0.8, 1.05, 0.05)
                plt.yticks(y_ticks, fontsize=24)
            plt.legend(loc=locs[i], labelspacing=0.2, prop={"size": 20})

            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

            if param_name == 'dev_ol':
                plt.xscale('log', base=2)

            if not os.path.exists(f'fig/shift'):
                os.makedirs(f'fig/shift')

            plt.savefig(f"fig/shift/{param_name}_{metric}.png") 


def draw_shiftest():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    param_name = 'eta'
    param = etas

    metrics = ['avg_shift_errs', 'avg_mu_errs', 'avg_sigma_errs']

    xlabel = r'Contamination Fraction $\eta$'
    methods = ['LME-OR', 'LME', 'MMLE', 'PQE']

    marks = ['-+', '-o', '-s', '-^']
    colors = ['limegreen', 'green', 'violet', 'orange']

    if not os.path.exists('fig/shiftest'):
        os.makedirs('fig/shiftest')


    for metric in metrics: 
        dir = f'res/shiftest/{param_name}/{metric}.csv'
        data = np.loadtxt(dir, delimiter=",", dtype=float)

        if metric == 'avg_shift_errs':
            print((data[:, 1] - data[:,0]) / data[:,1])
            print(np.mean((data[:, 1] - data[:,0]) / data[:,1]))

        plt.figure(figsize=figsize)

        ax = plt.gca()
        spine_lw = 2
        ax.spines['top'].set_linewidth(spine_lw)
        ax.spines['bottom'].set_linewidth(spine_lw)
        ax.spines['left'].set_linewidth(spine_lw)
        ax.spines['right'].set_linewidth(spine_lw)

        for j in range(len(methods)):
            plt.plot(param, data[:, j], marks[j], color=colors[j], lw=3, ms=18, label=methods[j])
        

        # plt.xlabel(xlabel, {"size": 32})
        plt.ylabel('Shift Error', {"size": 30})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(labelspacing=0.2, prop={"size": 20})

        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        if not os.path.exists(f'fig/shiftest/{param_name}'):
            os.makedirs(f'fig/shiftest/{param_name}')

        plt.savefig(f"fig/shiftest/{param_name}/{metric}.png") 


def draw_normal():
    ns = np.array([32, 64, 128, 256, 512])
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    dev_ols = np.array([0.25, 0.5, 1, 2, 4])

    params = [ns, etas, dev_ols]
    names = ['n', 'eta', 'dev']
    xlabels = [r'Dimensions $n$', r'Contamination Fraction $\eta$', r'Contamination Deviation $d$']
    methods = ['CM', 'GM', 'PM', 'PWM']

    marks = ['-d', '-s', '-o', '-x']
    colors = ['coral', 'royalblue', 'seagreen', 'limegreen']

    if not os.path.exists('fig/normal'):
        os.makedirs('fig/normal')

    # error rate
    for i, name in enumerate(names):
        dir = f'res/normal/{name}/mean_err.csv'
        data = np.loadtxt(dir, delimiter=",", dtype=float)

        # if name == 'eta':
        print((data[:, 2] - data[:,3]) / data[:,2])
        print(np.mean((data[:, 2] - data[:,3]) / data[:,2]))

        plt.figure(figsize=figsize)

        ax = plt.gca()
        spine_lw = 2
        ax.spines['top'].set_linewidth(spine_lw)
        ax.spines['bottom'].set_linewidth(spine_lw)
        ax.spines['left'].set_linewidth(spine_lw)
        ax.spines['right'].set_linewidth(spine_lw)

        for j in range(len(methods)):
            plt.plot(params[i], data[:, j], marks[j], color=colors[j], lw=3, ms=18, label=methods[j])

        # plt.xlabel(xlabels[i], {"size": 32})
        plt.ylabel('L2 Mean Error', {"size": 30})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(labelspacing=0.2, prop={"size": 20})

        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        if name != 'eta':
            plt.xscale('log', base=2)

        plt.savefig(f"fig/normal/mean_err_{name}.png") 


def draw_ar():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    dev_ols = np.array([0.25, 0.5, 1, 2, 4])
    ps = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    params = [etas, dev_ols, ps]
    param_names = ['eta', 'dev_ol', 'border']

    metric = 'target_ar_errs'

    xlabels = [r'Contamination Fraction $\eta$', r'Contamination Deviation $d$', r'Percentile $p$']

    methods = ['LME-OR', 'SME', 'LME', 'MMLE', 'PQE']

    marks = ['-+', '-*', '-o', '-s', '-^']
    colors = ['limegreen', 'aqua', 'green', 'violet', 'orange']

    if not os.path.exists('fig/ar'):
        os.makedirs('fig/ar')

    for i in range(len(params)):
        param_name = param_names[i]
        param = params[i]
        xlabel = xlabels[i]

        dir = f'res/ar/{param_name}/avg_{metric}.csv'
        data = np.loadtxt(dir, delimiter=",", dtype=float)

        plt.figure(figsize=figsize)

        ax = plt.gca()
        spine_lw = 2
        ax.spines['top'].set_linewidth(spine_lw)
        ax.spines['bottom'].set_linewidth(spine_lw)
        ax.spines['left'].set_linewidth(spine_lw)
        ax.spines['right'].set_linewidth(spine_lw)

        if i == 2:
            plt.plot(param, param, '--r', lw=3, label='Actual')
        else:
            plt.plot(param, np.ones(len(param)) * 0.5, '--r', lw=3, label='Actual')
        
        for j in range(len(methods)):
            plt.plot(param, data[:, j], marks[j], color=colors[j], lw=3, ms=18, label=methods[j])
        
        # plt.legend(ncol=3, loc=2 bbox_to_anchor=(0, 1.25), borderaxespad=0, frameon=False, labelspacing=0.2, columnspacing=0.2, prop={'size': 34})

        # plt.xlabel(xlabel, {"size": 32})
        plt.ylabel('Estimated Arrival Rate', {"size": 30})
        plt.xticks(fontsize=24)
        if i == 2:
            plt.yticks(fontsize=24)
        else:
            y_ticks = np.arange(0.4, 0.65, 0.05)
            plt.yticks(y_ticks, fontsize=24)
        plt.legend(ncol=2, labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        
        if param_name == 'dev_ol':
            plt.xscale('log', base=2)

        if not os.path.exists(f'fig/ar'):
            os.makedirs(f'fig/ar')

        plt.savefig(f"fig/ar/{param_name}_{metric}.png")

        if i == 2:
            # compute mean
            print(np.mean(np.abs(data[:,1] - ps), axis=0))


def draw_ar_t():
    ns = np.array([32, 64, 128, 256, 512])

    param = ns
    param_name = 'n'

    xlabel = r'Dimensions $n$'

    methods = ['LME-OR', 'SME', 'LME', 'MMLE', 'PQE']

    marks = ['-+', '-*', '-o', '-s', '-^']
    colors = ['limegreen', 'aqua', 'green', 'violet', 'orange']

    if not os.path.exists('fig/ar'):
        os.makedirs('fig/ar')

    dir = f'res/ar/{param_name}/times.csv'
    data = np.loadtxt(dir, delimiter=",", dtype=float)

    plt.figure(figsize=figsize)

    ax = plt.gca()
    spine_lw = 2
    ax.spines['top'].set_linewidth(spine_lw)
    ax.spines['bottom'].set_linewidth(spine_lw)
    ax.spines['left'].set_linewidth(spine_lw)
    ax.spines['right'].set_linewidth(spine_lw)
    
    for j in range(len(methods)):
        plt.plot(param, data[:, j], marks[j], color=colors[j], lw=3, ms=18, label=methods[j])

    # plt.xlabel(xlabel, {"size": 32})
    plt.ylabel('Running Time (s)', {"size": 30})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(ncol=2, labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    
    plt.xscale('log', base=2)

    if not os.path.exists(f'fig/ar'):
        os.makedirs(f'fig/ar')

    plt.savefig(f"fig/ar/{param_name}_times.png") 


def draw_intime():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ts = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    params = [ts, etas]

    names = ['t_rho', 'eta_thre']
    
    metrics = ['reward_ratio', 'thresholds']
    ylabels = [r'Untimely MEV Ratio', r'Most Profitable Delay']

    xlabels = [r'Delay $t$', r'Contamination Fraction $\eta$']
    methods = ['OIM', 'FM', 'RM', 'InTime']
    marks = ['-o', '-+', '-x', '-*']
    colors = [ 'C3', 'C0', 'C1', 'C2']


    if not os.path.exists('fig/intime'):
        os.makedirs('fig/intime')

    # error rate
    for i, name in enumerate(names):
        dir = f'res/intime/{name}.csv'
        # dir = f'res/intime/{param_name}/{metric}.csv'
        data = np.loadtxt(dir, delimiter=",", dtype=float)

        plt.figure(figsize=figsize)

        ax = plt.gca()
        spine_lw = 2
        ax.spines['top'].set_linewidth(spine_lw)
        ax.spines['bottom'].set_linewidth(spine_lw)
        ax.spines['left'].set_linewidth(spine_lw)
        ax.spines['right'].set_linewidth(spine_lw)

        for j in range(len(methods)):
            plt.plot(params[i], data[:, j], marks[j], color=colors[j], lw=3, ms=18, label=methods[j])
        
        # plt.legend(ncol=3, loc='', bbox_to_anchor=(0, 1.25), borderaxespad=0, frameon=False, labelspacing=0.2, columnspacing=0.2, prop={'size': 34})

        # plt.xlabel(xlabels[i], {"size": 32})
        plt.ylabel(ylabels[i], {"size": 30})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        # if name == 'eta_thre':
        #     plt.ylim(top=10)

        if not os.path.exists(f'fig/intime'):
            os.makedirs(f'fig/intime')

        plt.savefig(f"fig/intime/{name}.png") 


def draw_predict_():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    param = etas
    name = 'eta_pred'
    
    metrics = ['latency variance', 'throughput']
    ylabels = [r'Untimely MEV Ratio', 'Compatible Threshold']

    xlabel = r'Contamination Fraction $\eta$'
    methods = ['FM', 'RM', 'InTime']
    marks = ['-b+', '-yx', '-g*']

    if not os.path.exists('fig/intime'):
        os.makedirs('fig/intime')

    # error rate

    dir = f'res/intime/{name}.csv'
    data = np.loadtxt(dir, delimiter=",", dtype=float)

    plt.figure(figsize=figsize)

    ax = plt.gca()
    spine_lw = 2
    ax.spines['top'].set_linewidth(spine_lw)
    ax.spines['bottom'].set_linewidth(spine_lw)
    ax.spines['left'].set_linewidth(spine_lw)
    ax.spines['right'].set_linewidth(spine_lw)

    for j in range(len(methods)):
        plt.plot(param, data[:, j], marks[i][j], lw=3, ms=18, label=methods[j])
    
    plt.legend(ncol=3, loc=2, bbox_to_anchor=(0, 1.25), borderaxespad=0, frameon=False, labelspacing=0.2, columnspacing=0.2, prop={'size': 34})

    # plt.xlabel(xlabels[i], {"size": 32})
    plt.ylabel(ylabels, {"size": 30})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(labelspacing=0.2, prop={"size": 20})

    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)



    if not os.path.exists(f'fig/intime'):
        os.makedirs(f'fig/intime')

    plt.savefig(f"fig/intime/{name}.png") 


def draw_predict():
    # Sample data generation
    np.random.seed(0)
    data = {
        'Parameter': ['A'] * 30 + ['B'] * 30 + ['C'] * 30 + ['D'] * 30,
        'Method': ['Method1'] * 10 + ['Method2'] * 10 + ['Method3'] * 10 + 
                ['Method1'] * 10 + ['Method2'] * 10 + ['Method3'] * 10 +
                ['Method1'] * 10 + ['Method2'] * 10 + ['Method3'] * 10 +
                ['Method1'] * 10 + ['Method2'] * 10 + ['Method3'] * 10,
        'Value': np.random.normal(0, 1, 120),
        'LineValue': np.random.normal(1.5, 0.5, 120)
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Melt the DataFrame for easier plotting with seaborn
    df_melted = df.melt(id_vars=['Parameter', 'Method'], value_vars=['Value'], 
                        var_name='Metric', value_name='Value')

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Box plot
    sns.boxplot(x='Parameter', y='Value', hue='Method', data=df_melted, ax=ax1)
    ax1.set_title('Comparison of Methods Across Different Parameters')
    ax1.legend(title='Method', loc='upper left')

    # Creating a secondary axis
    ax2 = ax1.twinx()

    # Line plot data preparation
    line_data = df.groupby(['Parameter', 'Method'])['LineValue'].mean().reset_index()

    # Plotting lines for each method
    methods = line_data['Method'].unique()
    colors = ['black', 'red', 'blue']
    for method, color in zip(methods, colors):
        method_data = line_data[line_data['Method'] == method]
        ax2.plot(method_data['Parameter'], method_data['LineValue'], color=color, marker='o', linestyle='-', label=f'Line {method}')

    # Setting secondary axis labels and legend
    ax2.set_ylabel('Line Data Values')
    ax2.legend(loc='upper right')

    plt.show()


def draw_compress():
    # 读取数据
    data = np.loadtxt('res/overhead/compress.csv', delimiter=',', skiprows=1, usecols=(0,1,2,4,5))
    ns = data[:, 0].astype(int)
    original_size = data[:, 1] / 1024
    compressed_size = data[:, 2] / 1024
    compress_time = data[:, 3] * 1000  # 转换为毫秒
    decompress_time = data[:, 4] * 1000  # 转换为毫秒

    att_size = 723 / 1024  # 转换为 KB
    
    # 定义新的颜色方案
    royal_blue = '#4169E1'      # 皇家蓝
    dark_orange = '#FF8C00'     # 深橙色
    teal = '#008080'            # 青色
    gold = '#FFD700'            # 金色
    purple = '#800080'          # 紫色 (新颜色，替换森林绿)

    # 绘制大小比较图
    plt.figure(figsize=figsize)

    ax = plt.gca()
    spine_lw = 2
    ax.spines['top'].set_linewidth(spine_lw)
    ax.spines['bottom'].set_linewidth(spine_lw)
    ax.spines['left'].set_linewidth(spine_lw)
    ax.spines['right'].set_linewidth(spine_lw)
    
    x = np.arange(len(ns))
    width = 0.25  # 减小宽度以适应新的柱子

    rects1 = ax.bar(x - width, original_size, width, label='Raw Vector', color=royal_blue)
    rects2 = ax.bar(x, compressed_size, width, label='Compressed Vector', color=dark_orange)
    rects3 = ax.bar(x + width, [att_size] * len(ns), width, label='Attestation', color=purple)

    plt.ylabel('Message Size (KB)', {"size": 30})
    plt.xticks(x, ns, fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

    plt.tight_layout()
    plt.savefig("fig/overhead/compress_size.png")
    plt.close()

    # 绘制运行时间比较图
    plt.figure(figsize=figsize)

    ax = plt.gca()
    spine_lw = 2
    ax.spines['top'].set_linewidth(spine_lw)
    ax.spines['bottom'].set_linewidth(spine_lw)
    ax.spines['left'].set_linewidth(spine_lw)
    ax.spines['right'].set_linewidth(spine_lw)
    
    rects1 = ax.bar(x - width/2, compress_time, width, label='Compress', color=teal)
    rects2 = ax.bar(x + width/2, decompress_time, width, label='Decompress', color=gold)

    plt.ylabel('Running Time (ms)', {"size": 30})
    plt.xticks(x, ns, fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

    plt.tight_layout()
    plt.savefig("fig/overhead/compress_time.png")
    plt.close()

    # 计算并打印平均压缩和解压缩时间
    avg_compress_time = np.mean(compress_time)
    avg_decompress_time = np.mean(decompress_time)
    print(f"平均压缩时间: {avg_compress_time:.2f} 毫秒")
    print(f"平均解压缩时间: {avg_decompress_time:.2f} 毫秒")


def print_compress_usage():
    # 读取 CSV 文件
    df = pd.read_csv('res/overhead/compress_usage.csv')

    # 计算平均值
    avg_compression_cpu = np.mean(df['Compression CPU Usage (%)'])
    avg_compression_mem = np.mean(df['Compression Memory Usage (KB)'])
    avg_decompression_cpu = np.mean(df['Decompression CPU Usage (%)'])
    avg_decompression_mem = np.mean(df['Decompression Memory Usage (KB)'])

    # 打印结果
    print(f"压缩平均 CPU 使用率: {avg_compression_cpu:.2f}%")
    print(f"压缩平均内存使用: {avg_compression_mem:.2f} KB")
    print(f"解压平均 CPU 使用率: {avg_decompression_cpu:.2f}%")
    print(f"解压平均内存使用: {avg_decompression_mem:.2f} KB")


def draw_performance():
    # 读取数据
    data = np.loadtxt('res/overhead/performance.csv', delimiter=',', skiprows=1)
    ms = data[:, 0].astype(int)
    original_thr = data[:, 1]
    intime_thr = data[:, 2]
    original_lat = data[:, 3]
    intime_lat = data[:, 4]

    # 计算平均值
    avg_original_thr = np.mean(original_thr)
    avg_intime_thr = np.mean(intime_thr)
    avg_original_lat = np.mean(original_lat)
    avg_intime_lat = np.mean(intime_lat)

    # 打印平均值
    print(f"原始平均吞吐量: {avg_original_thr:.2f} tps")
    print(f"InTime平均吞吐量: {avg_intime_thr:.2f} tps")
    print(f"原始平均延迟: {avg_original_lat:.2f} s")
    print(f"InTime平均延迟: {avg_intime_lat:.2f} s")

    # 获取原始的 C2 和 C3 颜色
    c2_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    c3_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]

    # 将颜色调暗约 20%
    c2_darker = mcolors.rgb_to_hsv(mcolors.to_rgb(c2_color))
    c2_darker[2] *= 0.8  # 减少亮度
    c2_darker = mcolors.hsv_to_rgb(c2_darker)

    c3_darker = mcolors.rgb_to_hsv(mcolors.to_rgb(c3_color))
    c3_darker[2] *= 0.8  # 减少亮度
    c3_darker = mcolors.hsv_to_rgb(c3_darker)
    
    # 绘制大小比较图
    plt.figure(figsize=figsize)

    ax = plt.gca()
    spine_lw = 2
    ax.spines['top'].set_linewidth(spine_lw)
    ax.spines['bottom'].set_linewidth(spine_lw)
    ax.spines['left'].set_linewidth(spine_lw)
    ax.spines['right'].set_linewidth(spine_lw)
    
    x = np.arange(len(ms))
    width = 0.35

    rects1 = ax.bar(x - width/2, original_thr, width, label='Vanilla', color=c3_darker)
    rects2 = ax.bar(x + width/2, intime_thr, width, label='InTime', color=c2_darker)

    plt.ylabel('Throughput (tps)', {"size": 30})
    plt.xticks(x, ms, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(top=20)

    plt.legend(labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

    plt.tight_layout()
    plt.savefig("fig/overhead/overhead_thr.png")
    plt.close()

    # 绘制运行时间比较图
    plt.figure(figsize=figsize)

    ax = plt.gca()
    spine_lw = 2
    ax.spines['top'].set_linewidth(spine_lw)
    ax.spines['bottom'].set_linewidth(spine_lw)
    ax.spines['left'].set_linewidth(spine_lw)
    ax.spines['right'].set_linewidth(spine_lw)
    
    rects1 = ax.bar(x - width/2, original_lat, width, label='Vanilla', color=c3_darker)
    rects2 = ax.bar(x + width/2, intime_lat, width, label='InTime', color=c2_darker)

    plt.ylabel('Transaction Latency (s)', {"size": 30})
    plt.xticks(x, ms, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(top=20)

    plt.legend(labelspacing=0.2, columnspacing=0.2, prop={"size": 20})

    plt.tight_layout()
    plt.savefig("fig/overhead/overhead_lat.png")
    plt.close()


if __name__ == '__main__':
    # os.chdir(os.path.abspath('code/intime'))

    draw_shift()
    draw_shiftest()
    
    draw_normal()

    draw_ar()
    draw_ar_t()

    draw_intime()
    
    draw_predict()
    
    draw_compress()
    # print_compress_usage()
    
    draw_performance()












