# Makefile for HIcosmo English Documentation
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
	@echo "HIcosmo English Documentation Builder"
	@echo ""
	@echo "Available commands:"
	@echo "  make        - Build HTML documentation"
	@echo "  make html   - Build HTML documentation"
	@echo "  make clean  - Remove build directory"
	@echo "  make open   - Build and open in browser"
	@echo "  make help   - Show this help message"

html:
	@echo "Building documentation..."
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo ""
	@echo "Build complete. Documentation is in $(BUILDDIR)/html/"

clean:
	rm -rf $(BUILDDIR)
	@echo "Build directory cleaned"

open: html
	@echo "Opening documentation..."
	open $(BUILDDIR)/html/index.html
