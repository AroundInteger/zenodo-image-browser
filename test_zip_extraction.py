from src.api.zenodo import ZenodoAPI

# Zenodo record with a ZIP file for testing
record_id = "3518067"

zenodo = ZenodoAPI()
files = zenodo.get_files(record_id)

print(f"Files found in Zenodo record {record_id} (including ZIP contents):\n")
for f in files:
    if f.get("from_zip"):
        print(f"[ZIP] {f['key']} (source: {f['zip_source']}, inner path: {f['zip_inner_path']}, size: {f['size']} bytes)")
    else:
        print(f"      {f['key']} (size: {f.get('size', 'unknown')} bytes)") 