import shutil
from pathlib import Path
from fastapi import HTTPException, UploadFile
import os

# Base directory where graphs will be stored
GRAPHS_BASE_DIR = Path(os.getenv("GRAPHS_BASE_DIR", "/usr/data/graphs"))

# Ensure base directory exists
GRAPHS_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pt"}

# Max file size (e.g., 500MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

class FileHandler:
    @staticmethod
    def validate_file(filename: str) -> bool:
        """Check if file has allowed extension."""
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

    @staticmethod
    def create_graph_directory(graph_id: str) -> Path:
        """Create a new directory for the graph."""
        graph_dir = GRAPHS_BASE_DIR / graph_id
        graph_dir.mkdir(parents=True, exist_ok=True)
        return graph_dir

    @staticmethod
    async def save_uploaded_file(upload_file: UploadFile, destination: Path) -> None:
        """Save an uploaded file to destination."""
        file_size = 0
        
        try:
            contents = await upload_file.read()  # ← Read all content at once
            file_size = len(contents)
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                )
            
            # Write to file
            with open(destination, "wb") as buffer:
                buffer.write(contents)
                
        except HTTPException:
            raise
        except Exception as e:
            if destination.exists():
                destination.unlink()
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    @staticmethod
    def get_graph_folder_path(graph_id: str) -> str:
        """Get the full path where graph files are stored."""
        return str(GRAPHS_BASE_DIR / graph_id)

    @staticmethod
    def delete_graph_directory(graph_id: str) -> bool:
        """Delete the graph directory and all its contents."""
        graph_dir = GRAPHS_BASE_DIR / graph_id
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
            return True
        return False

    @staticmethod
    def list_graph_files(graph_id: str) -> list[str]:
        """List all .pt files in a graph directory."""
        graph_dir = GRAPHS_BASE_DIR / graph_id
        if not graph_dir.exists():
            return []
        
        return [
            f.name for f in graph_dir.glob("*.pt")
        ]

    @staticmethod
    def generate_random_hex(length: int = 16) -> str:
        """Generate a random hexadecimal string."""
        return os.urandom(length // 2).hex()