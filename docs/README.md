Building documentation
======================

Run the following commands from `docs` directory.

Automatic generate of Sphinx sources using [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)

```bash
make apidoc
```

This command only applies to newly created modules. It will not update modules that already exist. You will have to modify `docs/pysolotools.module_name` manually.

To build html files, run

```bash
make html
```

You can browse the documentation by opening `build/html/index.html` file directly in any web browser.

Cleanup build html files

```bash
make clean
```
