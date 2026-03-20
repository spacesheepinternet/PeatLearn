"""
Check and validate raw data files
"""
from pathlib import Path
import json

def check_raw_data():
    """Scan raw data directory and report what's there"""
    raw_dir = Path('data/raw')
    
    if not raw_dir.exists():
        print("❌ data/raw directory doesn't exist!")
        print("Creating it now...")
        raw_dir.mkdir(parents=True)
        return
    
    print("📁 Scanning data/raw directory...")
    print("="*60)
    
    # Count by file type
    file_types = {}
    total_size = 0
    all_files = []
    
    for file_path in raw_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size_mb': 0, 'files': []}
            
            file_types[ext]['count'] += 1
            file_types[ext]['size_mb'] += size_mb
            file_types[ext]['files'].append(str(file_path))
            
            total_size += size_mb
            all_files.append({
                'path': str(file_path),
                'size_mb': round(size_mb, 2),
                'type': ext
            })
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"Total files: {len(all_files)}")
    print(f"Total size: {total_size:.2f} MB")
    
    print(f"\n📄 By File Type:")
    for ext, info in sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"  {ext or '[no extension]'}: {info['count']} files ({info['size_mb']:.2f} MB)")
    
    # Show first few files of each type
    print(f"\n📋 Sample Files:")
    for ext, info in sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"\n  {ext or '[no extension]'} files:")
        for file_path in info['files'][:5]:  # Show first 5
            rel_path = Path(file_path).relative_to(raw_dir)
            print(f"    - {rel_path}")
        if len(info['files']) > 5:
            print(f"    ... and {len(info['files']) - 5} more")
    
    # Check for supported formats
    supported = ['.txt', '.pdf', '.docx', '.md', '.json', '.html']
    unsupported = [ext for ext in file_types.keys() if ext not in supported and ext]
    
    if unsupported:
        print(f"\n⚠️  Unsupported file types found:")
        for ext in unsupported:
            print(f"  {ext}: {file_types[ext]['count']} files")
        print(f"\nSupported formats: {', '.join(supported)}")
    
    # Save report
    report = {
        'total_files': len(all_files),
        'total_size_mb': round(total_size, 2),
        'by_type': {
            ext: {
                'count': info['count'],
                'size_mb': round(info['size_mb'], 2)
            }
            for ext, info in file_types.items()
        },
        'all_files': all_files
    }
    
    with open('data/artifacts/data_inventory.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: data/artifacts/data_inventory.json")
    
    # Next steps
    if len(all_files) > 0:
        print(f"\n🚀 Next Steps:")
        print(f"1. Run: python preprocessing/optimized_pipeline.py --input data/raw/[your_folder]")
        print(f"2. Or process everything: python preprocessing/optimized_pipeline.py --input data/raw")
    else:
        print(f"\n📝 Add your files to data/raw/ and run this script again")


if __name__ == "__main__":
    check_raw_data()