#!/usr/bin/env python3
"""
SmartSupport - Build Custom Knowledge Base
User-Friendly Document Upload Tool

This tool allows users to upload their own documents and build a custom knowledge base.
Documents are ingested into the Endee vector database and made searchable.

Usage:
    python build_knowledge_base.py          # Interactive mode
    python build_knowledge_base.py --folder ./documents     # Load all .txt files from folder
    python build_knowledge_base.py --file document.txt      # Load single file
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

from src.rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentUploader:
    """Interactive document upload utility for building knowledge bases."""
    
    def __init__(self):
        self.rag = RAGEngine()
        self.documents: List[Tuple[str, str]] = []
    
    def load_file(self, file_path: str) -> Tuple[str, str] | None:
        """Load a single text file and return (filename, content)."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return None
            
            if not path.suffix.lower() in ['.txt', '.md']:
                logger.warning(f"⚠️ File '{path.name}' is not .txt or .md, skipping...")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"⚠️ File '{path.name}' is empty, skipping...")
                return None
            
            logger.info(f"✅ Loaded: {path.name} ({len(content)} characters)")
            return (path.name, content)
        
        except Exception as e:
            logger.error(f"❌ Error loading file '{file_path}': {e}")
            return None
    
    def load_folder(self, folder_path: str) -> List[Tuple[str, str]]:
        """Load all .txt and .md files from a folder."""
        try:
            path = Path(folder_path)
            if not path.is_dir():
                logger.error(f"❌ Folder not found: {folder_path}")
                return []
            
            documents = []
            files = list(path.glob('*.txt')) + list(path.glob('*.md'))
            
            if not files:
                logger.warning(f"⚠️ No .txt or .md files found in '{folder_path}'")
                return []
            
            logger.info(f"📁 Found {len(files)} document(s) in '{folder_path}'")
            
            for file_path in files:
                result = self.load_file(str(file_path))
                if result:
                    documents.append(result)
            
            return documents
        
        except Exception as e:
            logger.error(f"❌ Error loading folder '{folder_path}': {e}")
            return []
    
    def interactive_mode(self):
        """Interactive menu for users to upload documents one by one."""
        print("\n" + "="*65)
        print(" SmartSupport - Build Your Knowledge Base (Interactive Mode)")
        print("="*65)
        print("\nOptions:")
        print("  1) Upload a single document file")
        print("  2) Upload all documents from a folder")
        print("  3) Exit without uploading\n")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            self._upload_single_interactive()
        elif choice == "2":
            self._upload_folder_interactive()
        elif choice == "3":
            print("❌ Cancelled.")
            return
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")
            return self.interactive_mode()
    
    def _upload_single_interactive(self):
        """Interactive single file upload."""
        print("\n" + "-"*65)
        print("Upload Single Document")
        print("-"*65)
        
        while True:
            file_path = input("\nEnter the path to your document (.txt or .md): ").strip()
            
            if not file_path:
                print("❌ Empty path. Please try again.")
                continue
            
            result = self.load_file(file_path)
            if result:
                self.documents.append(result)
                print(f"✅ Document '{result[0]}' added to upload queue.\n")
                
                another = input("Upload another document? (y/n): ").strip().lower()
                if another != 'y':
                    break
            else:
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    break
        
        if self.documents:
            self._ingest_documents()
    
    def _upload_folder_interactive(self):
        """Interactive folder upload."""
        print("\n" + "-"*65)
        print("Upload Documents from Folder")
        print("-"*65)
        
        folder_path = input("\nEnter the path to your documents folder: ").strip()
        
        if not folder_path:
            print("❌ Empty path.")
            return
        
        documents = self.load_folder(folder_path)
        if documents:
            self.documents = documents
            self._ingest_documents()
        else:
            print("❌ No documents found in folder.")
    
    def _ingest_documents(self):
        """Ingest all documents in the queue to Endee."""
        if not self.documents:
            print("❌ No documents to ingest.")
            return
        
        print("\n" + "="*65)
        print(f" Ingesting {len(self.documents)} document(s) into Knowledge Base")
        print("="*65 + "\n")
        
        total_chunks = 0
        successful = 0
        failed = 0
        
        for filename, content in self.documents:
            try:
                result = self.rag.ingest_text(content, filename=filename)
                num_chunks = result["num_chunks"]
                total_chunks += num_chunks
                successful += 1
                
                print(f"✅ {filename}")
                print(f"   └─ {num_chunks} chunks ingested")
                print(f"   └─ Document ID: {result['doc_id']}\n")
            
            except Exception as e:
                failed += 1
                logger.error(f"❌ Failed to ingest '{filename}': {e}\n")
        
        # Summary
        print("="*65)
        print(" Upload Summary")
        print("="*65)
        print(f"✅ Successful: {successful}/{len(self.documents)}")
        print(f"❌ Failed: {failed}/{len(self.documents)}")
        print(f"📊 Total chunks ingested: {total_chunks}\n")
        
        if successful > 0:
            print("🎉 Your knowledge base is ready!")
            print("   Open 'chat.html' in your browser to start asking questions.\n")


def cli_mode():
    """Command-line argument parsing for batch operations."""
    parser = argparse.ArgumentParser(
        description="SmartSupport - Build Custom Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_knowledge_base.py                          # Interactive mode
  python build_knowledge_base.py --folder ./documents     # Load all docs from folder
  python build_knowledge_base.py --folder ./docs --file manual.txt  # Both folder and file
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing documents to upload'
    )
    parser.add_argument(
        '--file',
        type=str,
        action='append',
        help='Path to single document file (can be used multiple times)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        args = cli_mode()
        uploader = DocumentUploader()
        
        # Load from folder if specified
        if args.folder:
            documents = uploader.load_folder(args.folder)
            uploader.documents.extend(documents)
        
        # Load individual files if specified
        if args.file:
            for file_path in args.file:
                result = uploader.load_file(file_path)
                if result:
                    uploader.documents.append(result)
        
        # Ingest all collected documents
        if uploader.documents:
            uploader._ingest_documents()
        else:
            print("❌ No documents found to ingest.")
            sys.exit(1)
    else:
        # Interactive mode
        uploader = DocumentUploader()
        uploader.interactive_mode()


if __name__ == "__main__":
    main()
