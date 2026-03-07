
import fsspec

def get_filesystem(path: str):
    if path.startswith("gs://"):
        import gcsfs
        return gcsfs.GCSFileSystem()
    else:
        return fsspec.filesystem("file")

def join_path(fs, *parts: str) -> str:
    if hasattr(fs, "sep"):
        sep = fs.sep
    else:
        sep = "/"
    return sep.join(parts)