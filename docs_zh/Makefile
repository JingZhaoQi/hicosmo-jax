# Makefile for HIcosmo Chinese Documentation
#
# Usage:
#   make        - Build HTML documentation
#   make html   - Build HTML documentation
#   make clean  - Remove build directory
#   make open   - Build and open in browser
#   make help   - Show available targets

SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html open

# Default target
all: html

help:
	@echo "HIcosmo 中文文档构建工具"
	@echo ""
	@echo "可用命令:"
	@echo "  make        - 构建 HTML 文档"
	@echo "  make html   - 构建 HTML 文档"
	@echo "  make clean  - 清除构建目录"
	@echo "  make open   - 构建并在浏览器中打开"
	@echo "  make help   - 显示此帮助信息"

html:
	@echo "正在构建文档..."
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo ""
	@echo "构建完成。文档位于 $(BUILDDIR)/html/"

clean:
	rm -rf $(BUILDDIR)
	@echo "已清除构建目录"

open: html
	@echo "正在打开文档..."
	open $(BUILDDIR)/html/index.html
