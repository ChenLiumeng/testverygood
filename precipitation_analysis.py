#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import rasterio
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def print_step_header(step_num, step_name):
    """打印步骤标题"""
    print("\n" + "=" * 60)
    print(f"步骤{step_num}: {step_name}")
    print("=" * 60)


def print_section_header(title):
    """打印章节标题"""
    print(f"\n{title}:")


def check_time_coords(data):
    """检查数据是否包含时间坐标"""
    if 'time' not in data.coords:
        print("数据中没有时间维度")
        return False, None
    
    time_coords = data.coords['time']
    if not hasattr(time_coords, 'dt'):
        print("时间坐标没有dt属性，无法提取时间信息")
        return False, None
    
    return True, time_coords


def get_valid_data(data):
    """获取有效数据（去除NaN）"""
    valid_data = data.values.flatten()
    return valid_data[~np.isnan(valid_data)]


def create_text_box(ax, text, facecolor='lightblue', alpha=0.8):
    """创建文本框"""
    ax.axis('off')
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor=facecolor, alpha=alpha))


def setup_plot_style(ax, title, xlabel, ylabel, grid=True, rotation=0):
    """设置图表样式"""
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if grid:
        ax.grid(True, alpha=0.3)
    if rotation != 0:
        ax.tick_params(axis='x', rotation=rotation)


def save_and_show_plot(filename, dpi=300):
    """保存并显示图表"""
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存为 '{filename}'")
    plt.show()


def filter_data_by_year(data, target_year=2018):
    """筛选指定年份的数据"""
    has_time, time_coords = check_time_coords(data)
    if not has_time:
        print(f"无法筛选{target_year}年数据：数据中没有时间维度")
        return data
    
    # 筛选指定年份的数据
    year_mask = time_coords.dt.year == target_year
    filtered_data = data.where(year_mask, drop=True)
    
    if filtered_data.sizes['time'] == 0:
        print(f"警告：{target_year}年没有数据，返回原始数据")
        return data
    
    print(f"已筛选出{target_year}年的数据，共{filtered_data.sizes['time']}个时间步")
    return filtered_data


def load_precipitation_data(file_path, target_year=2018):
    """
    加载NetCDF降水数据文件并筛选指定年份
    
    参数:
        file_path (str): NetCDF文件路径
        target_year (int): 目标年份，默认为2018
    
    返回:
        xarray.Dataset: 加载的数据集
    """
    print_step_header(1, f"加载NetCDF降水数据（筛选{target_year}年）")
    
    try:
        ds = xr.open_dataset(file_path)
        print(f"成功加载文件: {file_path}")
        
        # 查找降水变量
        precip_var = find_precipitation_variable(ds)
        data = ds[precip_var]
        
        # 筛选指定年份的数据
        filtered_data = filter_data_by_year(data, target_year)
        
        # 更新数据集
        ds[precip_var] = filtered_data
        
        return ds
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None


def print_data_structure(ds):
    """输出数据结构信息"""
    print_step_header(2, "数据结构信息")
    
    print_section_header("数据集基本信息")
    print(f"数据集名称: {ds.attrs.get('title', '未指定')}")
    print(f"数据描述: {ds.attrs.get('description', '未指定')}")
    
    print_section_header("维度信息")
    for dim_name, dim_size in ds.dims.items():
        print(f"  {dim_name}: {dim_size}")
    
    print_section_header("变量信息")
    for var_name, var in ds.data_vars.items():
        print(f"  {var_name}:")
        print(f"    形状: {var.shape}")
        print(f"    数据类型: {var.dtype}")
        print(f"    单位: {var.attrs.get('units', '未指定')}")
        print(f"    描述: {var.attrs.get('long_name', '未指定')}")
    
    print_section_header("坐标信息")
    for coord_name, coord in ds.coords.items():
        print(f"  {coord_name}:")
        if hasattr(coord, 'values'):
            if len(coord.values) > 5:
                print(f"    范围: {coord.values[0]} 到 {coord.values[-1]}")
                print(f"    长度: {len(coord.values)}")
            else:
                print(f"    值: {coord.values}")
    
    # 查找时间维度
    time_dims = [dim for dim in ds.dims.keys() if 'time' in dim.lower()]
    if time_dims:
        time_dim = time_dims[0]
        time_coord = ds[time_dim]
        print_section_header("时间范围")
        print(f"  开始时间: {time_coord.values[0]}")
        print(f"  结束时间: {time_coord.values[-1]}")
        print(f"  时间步数: {len(time_coord)}")


def find_precipitation_variable(ds):
    """查找降水变量"""
    precip_names = ['prcp', 'precipitation', 'rainfall', 'rain', 'pr', 'tp', 'precip']
    
    for var_name in ds.data_vars.keys():
        var_lower = var_name.lower()
        if any(name in var_lower for name in precip_names):
            return var_name
    
    # 如果没有找到典型的降水变量名，返回第一个变量
    return list(ds.data_vars.keys())[0]


def calculate_statistics(data):
    """计算基本统计量"""
    valid_data = get_valid_data(data)
    
    stats = {
        'min': np.nanmin(valid_data),
        'max': np.nanmax(valid_data),
        'mean': np.nanmean(valid_data),
        'std': np.nanstd(valid_data),
        'median': np.nanmedian(valid_data),
        'missing_count': np.isnan(data.values).sum(),
        'total_count': data.size,
        'missing_percent': (np.isnan(data.values).sum() / data.size) * 100,
        'shape': data.shape,
        'dims': data.dims
    }
    return stats


def print_statistics(stats, precip_var):
    """打印统计信息"""
    print(f"\n降水变量 '{precip_var}' 统计分析:")
    print(f"数据形状: {stats.get('shape', 'N/A')}")
    print(f"数据维度: {stats.get('dims', 'N/A')}")
    
    print_section_header("统计量")
    print(f"最小值: {stats['min']:.4f}")
    print(f"最大值: {stats['max']:.4f}")
    print(f"平均值: {stats['mean']:.4f}")
    print(f"标准差: {stats['std']:.4f}")
    print(f"中位数: {stats['median']:.4f}")
    
    print_section_header("缺失值统计")
    print(f"缺失值数量: {stats['missing_count']}")
    print(f"总数据量: {stats['total_count']}")
    print(f"缺失值百分比: {stats['missing_percent']:.2f}%")


def basic_statistics(ds, precip_var):
    """对降水变量进行基本统计分析"""
    print_step_header(3, "基本统计分析")
    
    data = ds[precip_var]
    stats = calculate_statistics(data)
    
    print_statistics(stats, precip_var)
    return data


# ==================== 可视化函数 ====================

def plot_time_series(data, ax, title="时间序列"):
    """绘制时间序列图"""
    time_series = data.mean(dim=[dim for dim in data.dims if dim != 'time'])
    time_coord = data.coords['time'] if 'time' in data.coords else range(len(time_series))
    
    ax.plot(time_coord, time_series, linewidth=1, alpha=0.8)
    setup_plot_style(ax, title, '时间', '降水量', rotation=45)


def plot_spatial_distribution(data, ax, time_idx=None, title="降水空间分布"):
    """绘制空间分布图"""
    if time_idx is None:
        time_idx = len(data.coords['time']) // 2 if 'time' in data.coords else 0
    
    spatial_data = data.isel(time=time_idx) if 'time' in data.coords else data
    
    im = ax.imshow(spatial_data.values, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='降水量')
    
    if 'lat' in spatial_data.coords and 'lon' in spatial_data.coords:
        setup_plot_style(ax, title, '经度', '纬度')
    else:
        setup_plot_style(ax, title, '网格索引 X', '网格索引 Y')


def plot_histogram(data, ax, title="数据分布直方图", bins=50, color='skyblue'):
    """绘制直方图"""
    valid_data = get_valid_data(data)
    ax.hist(valid_data, bins=bins, alpha=0.7, color=color, edgecolor='black')
    setup_plot_style(ax, title, '降水量', '频次')


def plot_monthly_boxplot(data, ax):
    """绘制月度箱线图"""
    has_time, time_coords = check_time_coords(data)
    
    if not has_time:
        create_text_box(ax, '无法生成月度箱线图\n(无时间维度)', 'lightcoral')
        setup_plot_style(ax, '各月降水量箱线图', '', '', grid=False)
        return
    
    monthly_data = []
    month_labels = []
    
    for month in range(1, 13):
        month_mask = time_coords.dt.month == month
        month_data = data.where(month_mask, drop=True).values.flatten()
        month_data = month_data[~np.isnan(month_data)]
        if len(month_data) > 0:
            monthly_data.append(month_data)
            month_labels.append(f'{month}月')
    
    if monthly_data:
        ax.boxplot(monthly_data, labels=month_labels)
        setup_plot_style(ax, '各月降水量箱线图', '月份', '降水量', rotation=45)
    else:
        create_text_box(ax, '无法生成月度箱线图\n(时间信息不足)', 'lightcoral')
        setup_plot_style(ax, '各月降水量箱线图', '', '', grid=False)


def plot_missing_data_timeline(data, ax):
    """绘制缺失值时间序列"""
    has_time, time_coords = check_time_coords(data)
    
    if not has_time:
        create_text_box(ax, '无法生成缺失值时间序列\n(无时间维度)', 'lightcoral')
        setup_plot_style(ax, '各时间步缺失值数量', '', '', grid=False)
        return
    
    missing_by_time = data.isnull().sum(dim=[dim for dim in data.dims if dim != 'time'])
    ax.plot(time_coords, missing_by_time, linewidth=1, alpha=0.8, color='red')
    setup_plot_style(ax, '各时间步缺失值数量', '时间', '缺失值数量', rotation=45)


def create_visualizations(ds, precip_var, data):
    """创建数据可视化图表"""
    print_step_header(4, "数据可视化")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 空间平均的降水时间序列图
    print("生成空间平均降水时间序列图...")
    ax1 = plt.subplot(2, 3, 1)
    plot_time_series(data, ax1, '空间平均降水时间序列')
    
    # 2. 某一时间点的降水空间分布图
    print("生成降水空间分布图...")
    ax2 = plt.subplot(2, 3, 2)
    plot_spatial_distribution(data, ax2)
    
    # 3. 降水数据的直方图
    print("生成降水直方图...")
    ax3 = plt.subplot(2, 3, 3)
    plot_histogram(data, ax3, '降水数据分布直方图')
    
    # 4. 各月降水量的箱线图
    print("生成月度降水箱线图...")
    ax4 = plt.subplot(2, 3, 4)
    plot_monthly_boxplot(data, ax4)
    
    # 5. 数据质量检查图
    print("生成数据质量检查图...")
    ax5 = plt.subplot(2, 3, 5)
    plot_missing_data_timeline(data, ax5)
    
    # 移除了统计摘要文本框
    
    save_and_show_plot('precipitation_analysis.png')


# ==================== 夏季降水分析函数 ====================

def filter_seasonal_data(data, start_month=6, end_month=9):
    """筛选季节性数据"""
    has_time, time_coords = check_time_coords(data)
    if not has_time:
        return None
    
    seasonal_mask = (time_coords.dt.month >= start_month) & (time_coords.dt.month <= end_month)
    seasonal_data = data.where(seasonal_mask, drop=True)
    
    if seasonal_data.sizes['time'] == 0:
        print("没有指定季节的数据，无法分析")
        return None
    
    return seasonal_data


def calculate_summer_precipitation(ds, precip_var):
    """计算夏季降水量"""
    print_step_header(5, "夏季降水分析")
    
    data = ds[precip_var]
    summer_data = filter_seasonal_data(data, 6, 9)
    
    if summer_data is None:
        return None
    
    # 计算夏季总降水量
    total_summer_precip = summer_data.sum(dim='time', skipna=True)
    
    # 创建单年数据结构，保持与其他函数兼容
    summer_precip_by_year = total_summer_precip.expand_dims(dim={'year': [2018]})
    summer_precip_by_year.coords['year'] = [2018]
    summer_precip_by_year.attrs['original_data'] = summer_data
    
    print(f"夏季总降水量计算完成")
    return summer_precip_by_year


def visualize_summer_precipitation(summer_data):
    """可视化夏季降水数据"""
    if summer_data is None:
        print("无法可视化夏季降水数据")
        return
    
    print("生成夏季降水可视化图表...")
    
    # 由于只有2018年数据，直接使用该年数据
    summer_precip_2018 = summer_data.sel(year=2018)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 夏季降水空间分布图
    ax1 = axes[0, 0]
    plot_spatial_distribution(summer_precip_2018, ax1, title='夏季降水量空间分布')
    
    # 2. 夏季降水直方图
    ax2 = axes[0, 1]
    plot_histogram(summer_precip_2018, ax2, '夏季降水量分布', bins=30, color='orange')
    
    # 3. 夏季各月降水量
    ax3 = axes[1, 0]
    # 获取原始数据来计算各月降水量
    original_data = summer_data.attrs.get('original_data', None)
    if original_data is not None:
        monthly_precip = []
        month_labels = []
        for month in [6, 7, 8, 9]:
            month_mask = original_data['time'].dt.month == month
            month_data = original_data.where(month_mask, drop=True)
            if month_data.sizes['time'] > 0:
                total_month_precip = month_data.sum(dim='time', skipna=True).values.flatten()
                total_month_precip = total_month_precip[~np.isnan(total_month_precip)]
                if len(total_month_precip) > 0:
                    monthly_precip.append(total_month_precip)
                    month_labels.append(f'{month}月')
        
        if monthly_precip:
            ax3.boxplot(monthly_precip, labels=month_labels)
            setup_plot_style(ax3, '夏季各月降水量', '月份', '降水量')
        else:
            create_text_box(ax3, '无法生成月度降水量图\n(数据不足)', 'lightcoral')
            setup_plot_style(ax3, '夏季各月降水量', '', '', grid=False)
    else:
        create_text_box(ax3, '无法生成月度降水量图\n(缺少原始数据)', 'lightcoral')
        setup_plot_style(ax3, '夏季各月降水量', '', '', grid=False)
    
    # 4. 移除了夏季降水量统计摘要
    
    save_and_show_plot('summer_precipitation_analysis.png')


# ==================== 耕地面积分析函数 ====================

def load_land_use_data(file_path):
    """加载耕地面积数据"""
    print_step_header(6, "加载耕地面积数据")
    
    try:
        with rasterio.open(file_path) as src:
            land_use_data = src.read(1)  # 读取第一个波段
            transform = src.transform
            crs = src.crs
            
        print(f"成功加载耕地面积数据: {file_path}")
        print(f"数据形状: {land_use_data.shape}")
        print(f"数据类型: {land_use_data.dtype}")
        print(f"坐标参考系统: {crs}")
        
        return land_use_data, transform, crs
    except Exception as e:
        print(f"加载耕地面积数据失败: {e}")
        return None, None, None

def analyze_land_use_data(land_use_data):
    """分析耕地面积数据"""
    print_section_header("耕地面积数据统计")
    
    # 统计不同土地利用类型
    unique_values, counts = np.unique(land_use_data, return_counts=True)
    
    print("土地利用类型统计:")
    for value, count in zip(unique_values, counts):
        percentage = (count / land_use_data.size) * 100
        print(f"  类型 {value}: {count} 像素 ({percentage:.2f}%)")
    
    # 假设1表示耕地，0表示非耕地（根据实际数据调整）
    # 这里需要根据实际数据格式来确定耕地和非耕地的值
    print("\n注意：请根据实际数据格式确认耕地和非耕地的像素值")
    
    return unique_values, counts

def visualize_land_use_data(land_use_data):
    """可视化耕地面积数据"""
    print("生成耕地面积分布图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 耕地面积空间分布图
    ax1 = axes[0]
    im1 = ax1.imshow(land_use_data, cmap='YlOrBr', aspect='auto')
    plt.colorbar(im1, ax=ax1, label='土地利用类型')
    setup_plot_style(ax1, '2013年耕地面积空间分布', '经度', '纬度')
    
    # 2. 土地利用类型统计图
    ax2 = axes[1]
    unique_values, counts = np.unique(land_use_data, return_counts=True)
    ax2.bar(unique_values, counts, alpha=0.7, color='green', edgecolor='black')
    setup_plot_style(ax2, '土地利用类型统计', '土地利用类型', '像素数量')
    
    save_and_show_plot('land_use_analysis.png')

def resample_data_to_match(precip_data, land_use_data, precip_shape):
    """将耕地数据重采样到与降水数据相同的分辨率"""
    print("重采样耕地数据以匹配降水数据分辨率...")
    
    # 简单的重采样方法：使用最近邻插值
    if land_use_data.shape != precip_shape:
        try:
            from scipy.ndimage import zoom
            
            # 计算缩放因子
            scale_y = float(precip_shape[0]) / float(land_use_data.shape[0])
            scale_x = float(precip_shape[1]) / float(land_use_data.shape[1])
            
            # 重采样
            resampled_land_use = zoom(land_use_data, (scale_y, scale_x), order=0)
            
            # 简单的裁剪到目标尺寸
            if resampled_land_use.shape[0] > precip_shape[0]:
                resampled_land_use = resampled_land_use[:precip_shape[0], :]
            if resampled_land_use.shape[1] > precip_shape[1]:
                resampled_land_use = resampled_land_use[:, :precip_shape[1]]
            
            print(f"重采样完成: {land_use_data.shape} -> {resampled_land_use.shape}")
            return resampled_land_use
        except Exception as e:
            print(f"重采样失败: {e}")
            print("使用原始数据")
            return land_use_data
    else:
        print("数据分辨率已匹配，无需重采样")
        return land_use_data

def classify_land_use(land_use_data):
    """分类耕地和非耕地"""
    # 这里需要根据实际数据格式来确定分类标准
    # 假设1表示耕地，其他值表示非耕地
    # 请根据实际数据调整这些值
    
    print("分类耕地和非耕地...")
    
    # 获取唯一值
    unique_values = np.unique(land_use_data)
    print(f"数据中的唯一值: {unique_values}")
    
    # 这里需要根据实际数据格式来确定哪些值代表耕地
    # 常见的耕地分类值可能是1, 2, 10等，具体需要查看数据说明
    
    # 示例：假设值为1的像素是耕地
    # 请根据实际数据调整这个条件
    if 1 in unique_values:
        farmland_mask = (land_use_data == 1)
        non_farmland_mask = (land_use_data != 1)
    else:
        # 如果没有值为1，使用第一个非零值作为耕地
        non_zero_values = unique_values[unique_values != 0]
        if len(non_zero_values) > 0:
            farmland_mask = (land_use_data == non_zero_values[0])
            non_farmland_mask = (land_use_data != non_zero_values[0])
        else:
            print("警告：未找到有效的耕地分类值")
            return None, None
    
    farmland_pixels = np.sum(farmland_mask)
    non_farmland_pixels = np.sum(non_farmland_mask)
    
    print(f"耕地像素数: {farmland_pixels}")
    print(f"非耕地像素数: {non_farmland_pixels}")
    print(f"耕地比例: {farmland_pixels / land_use_data.size * 100:.2f}%")
    
    return farmland_mask, non_farmland_mask

def extract_precipitation_by_land_use(summer_precip_data, farmland_mask, non_farmland_mask):
    """根据土地利用类型提取降水量数据"""
    print("根据土地利用类型提取降水量数据...")
    
    # 提取耕地和非耕地的降水量
    farmland_precip = summer_precip_data.values[farmland_mask]
    non_farmland_precip = summer_precip_data.values[non_farmland_mask]
    
    # 去除NaN值
    farmland_precip = farmland_precip[~np.isnan(farmland_precip)]
    non_farmland_precip = non_farmland_precip[~np.isnan(non_farmland_precip)]
    
    print(f"耕地降水量样本数: {len(farmland_precip)}")
    print(f"非耕地降水量样本数: {len(non_farmland_precip)}")
    
    return farmland_precip, non_farmland_precip

def perform_statistical_test(farmland_precip, non_farmland_precip):
    """执行统计显著性检验"""
    print_section_header("统计显著性检验")
    
    if len(farmland_precip) == 0 or len(non_farmland_precip) == 0:
        print("错误：样本数量不足，无法进行统计检验")
        return None
    
    # 基本统计量
    print("基本统计量:")
    print(f"耕地平均降水量: {np.mean(farmland_precip):.2f} ± {np.std(farmland_precip):.2f}")
    print(f"非耕地平均降水量: {np.mean(non_farmland_precip):.2f} ± {np.std(non_farmland_precip):.2f}")
    
    # 正态性检验
    print("\n正态性检验 (Shapiro-Wilk):")
    _, p_farmland = stats.shapiro(farmland_precip)
    _, p_non_farmland = stats.shapiro(non_farmland_precip)
    print(f"耕地降水量正态性 p值: {p_farmland:.4f}")
    print(f"非耕地降水量正态性 p值: {p_non_farmland:.4f}")
    
    # 方差齐性检验
    print("\n方差齐性检验 (Levene):")
    _, p_levene = stats.levene(farmland_precip, non_farmland_precip)
    print(f"方差齐性 p值: {p_levene:.4f}")
    
    # 根据正态性检验结果选择检验方法
    alpha = 0.05
    is_normal = (p_farmland > alpha) and (p_non_farmland > alpha)
    
    if is_normal:
        print("\n使用独立样本t检验:")
        t_stat, p_value = ttest_ind(farmland_precip, non_farmland_precip)
        test_name = "独立样本t检验"
    else:
        print("\n使用Mann-Whitney U检验:")
        u_stat, p_value = mannwhitneyu(farmland_precip, non_farmland_precip, alternative='two-sided')
        test_name = "Mann-Whitney U检验"
    
    print(f"检验方法: {test_name}")
    print(f"p值: {p_value:.6f}")
    
    # 判断显著性
    if p_value < alpha:
        print(f"结果: 存在显著差异 (p < {alpha})")
        if np.mean(farmland_precip) > np.mean(non_farmland_precip):
            print("结论: 耕地夏季降水量显著高于非耕地")
        else:
            print("结论: 耕地夏季降水量显著低于非耕地")
    else:
        print(f"结果: 无显著差异 (p >= {alpha})")
        print("结论: 耕地和非耕地夏季降水量无显著差异")
    
    return {
        'test_name': test_name,
        'p_value': p_value,
        'significant': p_value < alpha,
        'farmland_mean': np.mean(farmland_precip),
        'non_farmland_mean': np.mean(non_farmland_precip),
        'farmland_std': np.std(farmland_precip),
        'non_farmland_std': np.std(non_farmland_precip)
    }

def visualize_precipitation_comparison(farmland_precip, non_farmland_precip, test_results):
    """可视化耕地和非耕地降水量对比"""
    print("生成耕地和非耕地降水量对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 箱线图对比
    ax1 = axes[0, 0]
    data_to_plot = [farmland_precip, non_farmland_precip]
    labels = ['耕地', '非耕地']
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # 设置颜色
    colors = ['lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    setup_plot_style(ax1, '耕地vs非耕地夏季降水量对比', '土地利用类型', '夏季降水量')
    
    # 2. 直方图对比
    ax2 = axes[0, 1]
    ax2.hist(farmland_precip, bins=30, alpha=0.7, label='耕地', color='lightgreen', edgecolor='black')
    ax2.hist(non_farmland_precip, bins=30, alpha=0.7, label='非耕地', color='lightblue', edgecolor='black')
    ax2.legend()
    setup_plot_style(ax2, '耕地vs非耕地夏季降水量分布', '夏季降水量', '频次')
    
    # 3. 均值对比图
    ax3 = axes[1, 0]
    means = [test_results['farmland_mean'], test_results['non_farmland_mean']]
    stds = [test_results['farmland_std'], test_results['non_farmland_std']]
    bars = ax3.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['lightgreen', 'lightblue'], edgecolor='black')
    setup_plot_style(ax3, '耕地vs非耕地夏季降水量均值对比', '土地利用类型', '平均夏季降水量')
    
    # 4. 统计检验结果
    ax4 = axes[1, 1]
    test_text = f"""
统计检验结果:

检验方法: {test_results['test_name']}
p值: {test_results['p_value']:.6f}

均值对比:
耕地: {test_results['farmland_mean']:.2f} ± {test_results['farmland_std']:.2f}
非耕地: {test_results['non_farmland_mean']:.2f} ± {test_results['non_farmland_std']:.2f}

样本数量:
耕地: {len(farmland_precip)}
非耕地: {len(non_farmland_precip)}

显著性: {'显著' if test_results['significant'] else '不显著'}
    """
    create_text_box(ax4, test_text, 'lightyellow')
    
    save_and_show_plot('precipitation_land_use_comparison.png')

# ==================== 主函数 ====================

def main():
    """主函数：执行完整的降水数据分析流程"""
    print("NetCDF降水数据分析开始")
    print("=" * 60)
    
    file_path = "precipitation.nc"
    land_use_file = r"C:\Users\clm13\Desktop\try\2013_irr_lands.tif"
    target_year = 2018
    
    # 步骤1: 加载数据（筛选指定年份）
    ds = load_precipitation_data(file_path, target_year)
    if ds is None:
        return
    
    # 步骤2: 输出数据结构信息
    print_data_structure(ds)
    
    # 步骤3: 查找降水变量
    precip_var = find_precipitation_variable(ds)
    print(f"\n识别到的降水变量: {precip_var}")
    
    # 步骤4: 基本统计分析
    data = basic_statistics(ds, precip_var)
    
    # 步骤5: 数据可视化
    create_visualizations(ds, precip_var, data)
    
    # 步骤6: 夏季降水分析
    summer_data = calculate_summer_precipitation(ds, precip_var)
    visualize_summer_precipitation(summer_data)
    
    # 步骤7: 耕地面积分析
    land_use_data, transform, crs = load_land_use_data(land_use_file)
    if land_use_data is not None:
        # 分析耕地数据
        unique_values, counts = analyze_land_use_data(land_use_data)
        
        # 可视化耕地数据
        visualize_land_use_data(land_use_data)
        
        # 重采样耕地数据以匹配降水数据
        summer_precip_2018 = summer_data.sel(year=2018)
        resampled_land_use = resample_data_to_match(summer_precip_2018, land_use_data, summer_precip_2018.shape)
        
        if resampled_land_use is not None:
            # 分类耕地和非耕地
            farmland_mask, non_farmland_mask = classify_land_use(resampled_land_use)
            
            if farmland_mask is not None:
                # 提取不同土地利用类型的降水量
                farmland_precip, non_farmland_precip = extract_precipitation_by_land_use(
                    summer_precip_2018, farmland_mask, non_farmland_mask)
                
                # 执行统计检验
                test_results = perform_statistical_test(farmland_precip, non_farmland_precip)
                
                if test_results is not None:
                    # 可视化对比结果
                    visualize_precipitation_comparison(farmland_precip, non_farmland_precip, test_results)
    
    print("\n" + "=" * 60)
    print(f"{target_year}年降水数据分析完成！")
    print("=" * 60)
    print("生成的文件:")
    print("  - precipitation_analysis.png: 基本分析图表")
    print("  - summer_precipitation_analysis.png: 夏季降水分析图表")
    print("  - land_use_analysis.png: 耕地面积分析图表")
    print("  - precipitation_land_use_comparison.png: 耕地vs非耕地降水量对比图")
    print(f"\n分析内容包括:")
    print(f"  {target_year}年数据结构信息")
    print(f"  {target_year}年基本统计分析")
    print(f"  {target_year}年时间序列可视化")
    print(f"  {target_year}年空间分布图")
    print(f"  {target_year}年数据分布直方图")
    print(f"  {target_year}年月度箱线图")
    print(f"  {target_year}年夏季降水分析")
    print(f"  {target_year}年数据质量检查")
    print("  2013年耕地面积分析")
    print("  耕地vs非耕地夏季降水量差异分析")

if __name__ == "__main__":
    main()    