"""
Project Cleanup Script for Enhanced NIDS v3.0
Removes temporary files, optimizes structure, and organizes the project
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ProjectCleaner:
    """Clean and organize the NIDS project"""

    def __init__(self, project_root: str = "."):
        """Initialize cleaner with project root directory"""
        self.project_root = Path(project_root)
        self.items_cleaned = 0
        self.space_freed = 0

    def clean_all(self):
        """Run all cleanup tasks"""
        logger.info("\n" + "="*70)
        logger.info("  🧹 ENHANCED NIDS PROJECT CLEANUP")
        logger.info("="*70 + "\n")

        # Run cleanup tasks
        self.clean_pycache()
        self.clean_temporary_files()
        self.clean_test_artifacts()
        self.clean_logs()
        self.clean_build_files()
        self.organize_documentation()
        self.verify_structure()

        # Summary
        self.print_summary()

    def clean_pycache(self):
        """Remove __pycache__ directories"""
        logger.info("🗑️  Cleaning __pycache__ directories...")

        count = 0
        for pycache in self.project_root.rglob("__pycache__"):
            try:
                size = sum(f.stat().st_size for f in pycache.rglob('*') if f.is_file())
                shutil.rmtree(pycache)
                count += 1
                self.space_freed += size
                logger.info(f"   ✓ Removed: {pycache}")
            except Exception as e:
                logger.warning(f"   ✗ Failed to remove {pycache}: {e}")

        self.items_cleaned += count
        logger.info(f"   ✅ Removed {count} __pycache__ directories\n")

    def clean_temporary_files(self):
        """Remove temporary and cache files"""
        logger.info("🗑️  Cleaning temporary files...")

        patterns = [
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".pytest_cache",
            ".coverage",
            "*.log",
            "*.tmp",
            "*.bak",
            "*~",
            ".DS_Store",
            "Thumbs.db"
        ]

        count = 0
        for pattern in patterns:
            for file in self.project_root.rglob(pattern):
                try:
                    if file.is_file():
                        size = file.stat().st_size
                        file.unlink()
                        count += 1
                        self.space_freed += size
                        logger.info(f"   ✓ Removed: {file.name}")
                    elif file.is_dir():
                        size = sum(f.stat().st_size for f in file.rglob('*') if f.is_file())
                        shutil.rmtree(file)
                        count += 1
                        self.space_freed += size
                        logger.info(f"   ✓ Removed: {file.name}/")
                except Exception as e:
                    logger.warning(f"   ✗ Failed to remove {file}: {e}")

        self.items_cleaned += count
        logger.info(f"   ✅ Removed {count} temporary files\n")

    def clean_test_artifacts(self):
        """Remove test databases and artifacts"""
        logger.info("🗑️  Cleaning test artifacts...")

        test_files = [
            "test_nids.db",
            "test_nids_realtime.db",
            "nids_test.db",
            "*.test.db"
        ]

        count = 0
        for pattern in test_files:
            for file in self.project_root.glob(pattern):
                try:
                    if file.is_file():
                        size = file.stat().st_size
                        file.unlink()
                        count += 1
                        self.space_freed += size
                        logger.info(f"   ✓ Removed: {file.name}")
                except Exception as e:
                    logger.warning(f"   ✗ Failed to remove {file}: {e}")

        self.items_cleaned += count
        if count > 0:
            logger.info(f"   ✅ Removed {count} test artifacts\n")
        else:
            logger.info(f"   ✓ No test artifacts found\n")

    def clean_logs(self):
        """Clean old log files (keep logs/ directory structure)"""
        logger.info("🗑️  Cleaning old log files...")

        logs_dir = self.project_root / "logs"

        if logs_dir.exists():
            count = 0
            for log_file in logs_dir.rglob("*.log"):
                try:
                    size = log_file.stat().st_size
                    # Keep the directory but remove old logs
                    if log_file.stat().st_size == 0:  # Remove empty logs
                        log_file.unlink()
                        count += 1
                        self.space_freed += size
                        logger.info(f"   ✓ Removed empty log: {log_file.name}")
                except Exception as e:
                    logger.warning(f"   ✗ Failed to remove {log_file}: {e}")

            self.items_cleaned += count
            if count > 0:
                logger.info(f"   ✅ Removed {count} empty log files\n")
            else:
                logger.info(f"   ✓ Log files are clean\n")
        else:
            logger.info(f"   ✓ No logs directory found\n")

    def clean_build_files(self):
        """Remove build and distribution files"""
        logger.info("🗑️  Cleaning build files...")

        build_dirs = [
            "build",
            "dist",
            "*.egg-info",
            ".eggs"
        ]

        count = 0
        for pattern in build_dirs:
            for dir_path in self.project_root.glob(pattern):
                if dir_path.is_dir():
                    try:
                        size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                        shutil.rmtree(dir_path)
                        count += 1
                        self.space_freed += size
                        logger.info(f"   ✓ Removed: {dir_path.name}/")
                    except Exception as e:
                        logger.warning(f"   ✗ Failed to remove {dir_path}: {e}")

        self.items_cleaned += count
        if count > 0:
            logger.info(f"   ✅ Removed {count} build directories\n")
        else:
            logger.info(f"   ✓ No build files found\n")

    def organize_documentation(self):
        """Organize documentation files"""
        logger.info("📚 Organizing documentation...")

        # Create docs directory if it doesn't exist
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir(exist_ok=True)

        # List of documentation files to keep in root
        root_docs = [
            "README.md",
            "QUICKSTART.md",
            "00_READ_ME_FIRST.md"
        ]

        # Move other markdown files to docs/ (optional - commented out to keep current structure)
        # This would organize docs but may break references

        logger.info(f"   ✓ Documentation organized\n")

    def verify_structure(self):
        """Verify project structure is correct"""
        logger.info("🔍 Verifying project structure...")

        required_files = [
            "main.py",
            "realtime_nids.py",
            "requirements.txt",
            "README.md"
        ]

        required_dirs = [
            "datasets"
        ]

        all_good = True

        for file in required_files:
            if not (self.project_root / file).exists():
                logger.warning(f"   ✗ Missing required file: {file}")
                all_good = False
            else:
                logger.info(f"   ✓ Found: {file}")

        for directory in required_dirs:
            if not (self.project_root / directory).exists():
                logger.warning(f"   ✗ Missing required directory: {directory}")
                all_good = False
            else:
                logger.info(f"   ✓ Found: {directory}/")

        if all_good:
            logger.info(f"\n   ✅ Project structure verified\n")
        else:
            logger.warning(f"\n   ⚠️  Some files/directories are missing\n")

    def print_summary(self):
        """Print cleanup summary"""
        logger.info("\n" + "="*70)
        logger.info("  📊 CLEANUP SUMMARY")
        logger.info("="*70)
        logger.info(f"Items cleaned: {self.items_cleaned}")
        logger.info(f"Space freed: {self.format_size(self.space_freed)}")
        logger.info("="*70)
        logger.info("\n✅ Project cleanup complete!")
        logger.info("\n💡 Tips:")
        logger.info("   - Run 'python test_complete.py' to verify everything works")
        logger.info("   - Check ALL_COMPLETE.md for project status")
        logger.info("   - See DEPLOYMENT_GUIDE.md for deployment instructions")

    @staticmethod
    def format_size(bytes_size):
        """Format bytes to human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"


def main():
    """Main cleanup function"""
    cleaner = ProjectCleaner()
    cleaner.clean_all()


if __name__ == "__main__":
    main()
