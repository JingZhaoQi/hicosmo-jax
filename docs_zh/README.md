# HIcosmo 中文 Read the Docs

该目录用于构建 HIcosmo 的中文 Read the Docs 文档站点。

## 本地构建

```bash
pip install -r docs_zh/requirements.txt
sphinx-build -b html docs_zh/source docs_zh/build/html
```

## Read the Docs

根目录的 `.readthedocs.yml` 已指向本目录。
