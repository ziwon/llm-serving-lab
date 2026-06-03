# ONNX Viewer

Use Netron to inspect ONNX model graphs in a browser.

```bash
just onnx-view path/to/model.onnx
```

The default URL is:

```bash
http://127.0.0.1:8081
```

Use a different port if needed:

```bash
just onnx-view path/to/model.onnx 8082
```

The viewer runs in the foreground. Stop it with `Ctrl-C`.
