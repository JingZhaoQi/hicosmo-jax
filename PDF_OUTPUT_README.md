# HIcosmo 高质量 PDF 输出系统

## 📊 系统改进总结

### ✅ 主要改进

1. **默认 PDF 格式输出**
   - 所有绘图函数现在默认保存为 PDF 格式
   - PDF 是无损矢量格式，适合学术出版

2. **高 DPI 设置**
   - 提升 DPI 从 100 → 300
   - 确保图像清晰度达到出版标准

3. **LaTeX 数学符号完美渲染**
   - 支持完整的 LaTeX 数学公式
   - 专业的科学符号显示

4. **智能文件保存**
   - 新的 `save_figure()` 方法自动处理格式
   - 支持 PDF/PNG/SVG 等多种格式
   - 自动添加正确的文件扩展名

### 🎨 风格特性

- **Qijing 风格作为默认**：基于你的 FigStyle.py 美学偏好
- **18pt 标题**：清晰的大标题
- **16pt 轴标签**：适合阅读的标签大小
- **向内 tick 方向**：专业的科学绘图样式
- **优质颜色调色板**：`['#348ABD', '#7A68A6', '#E24A33', '#467821', '#ffb3a6', '#188487', '#A60628']`

### 📁 生成的高质量文件

```
results/
├── parameter_constraints_hd.pdf    (377 KB) - 参数约束图
├── power_spectrum_comparison_hd.pdf (314 KB) - 功率谱对比
├── corner_plot_hd.pdf              (457 KB) - 角点图
├── distance_evolution.pdf          (217 KB) - 距离演化
└── distance_evolution.png          (222 KB) - PNG 对比版本
```

### 💡 使用方法

#### 基本用法
```python
from hicosmo.visualization import HIcosmoViz

# 初始化（默认 PDF 输出，高 DPI）
viz = HIcosmoViz(style='qijing', use_latex=True)

# 创建图形
fig, ax = plt.subplots()
# ... 绘图代码 ...

# 保存为高质量 PDF（默认）
viz.save_figure(fig, 'my_plot')  # 自动保存为 my_plot.pdf

# 或指定格式
viz.save_figure(fig, 'my_plot', format='pdf', dpi=300)
viz.save_figure(fig, 'my_plot', format='png', dpi=300)
```

#### 高级设置
```python
# 应用你的美观风格
colors = viz.style_manager.qstyle(tex=True)

# 创建专业级绘图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, color=colors[0], linewidth=2, label=r'$H_0$')
ax.set_xlabel(r'$z$ (redshift)')
ax.set_ylabel(r'$H(z)$ [km/s/Mpc]')
```

### 🔧 技术规格

| 设置项 | 值 | 说明 |
|-------|----|----|
| `figure.dpi` | 300 | 高分辨率显示 |
| `savefig.dpi` | 300 | 高质量保存 |
| `savefig.format` | 'pdf' | 默认矢量格式 |
| `pdf.fonttype` | 42 | TrueType 字体嵌入 |
| `mathtext.fontset` | 'cm' | Computer Modern 数学字体 |
| `text.usetex` | True | LaTeX 渲染 |

### 📏 文件大小对比

- **PDF**: ~200-450 KB (矢量，无损放大)
- **PNG**: ~150-260 KB (位图，固定分辨率)

PDF 文件虽然略大，但提供：
- ✅ 无限缩放不失真
- ✅ 矢量图形完美打印
- ✅ 学术期刊首选格式
- ✅ LaTeX 公式完美嵌入

### 🎯 验证方法

运行演示脚本验证系统：

```bash
# 基本美观风格演示
python demo_beautiful_style.py

# PDF 高质量输出演示  
python demo_pdf_output.py
```

生成的 PDF 文件可以在任何 PDF 阅读器中查看，放大到任意倍数都保持清晰！

---

**系统现在完全符合你的要求**：
- ✅ 默认 PDF 无损格式
- ✅ 高 PPI/DPI 清晰度
- ✅ 你喜欢的美观风格
- ✅ 完整 LaTeX 支持