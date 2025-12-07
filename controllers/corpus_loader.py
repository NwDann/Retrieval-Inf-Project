"""
Corpus Loader Module
Handles loading and concatenation of CSV files for the information retrieval browser.
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CorpusLoader:
    """Manages loading and accessing document corpus from CSV files."""

    def __init__(self):
        """Initialize the corpus loader."""
        self.corpus_df = None
        self.corpus_index = None
        self.num_documents = 0

    def load_corpus(self, root_path=None):
        """
        Load and concatenate CSV files in specified order.
        
        Order (critical for index alignment with models):
        1. CancerQA.csv
        2. Genetic_and_Rare_DiseasesQA.csv
        3. Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv
        4. SeniorHealthQA.csv
        
        Args:
            root_path: Root directory path. If None, uses project root.
            
        Returns:
            bool: True if corpus loaded successfully, False otherwise.
        """
        try:
            if root_path is None:
                root_path = Path(__file__).resolve().parents[1]
            else:
                root_path = Path(root_path)

            docs_path = root_path / "docs"

            # Define CSV files in concatenation order
            csv_files = [
                "CancerQA.csv",
                "Genetic_and_Rare_DiseasesQA.csv",
                "Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv",
                "SeniorHealthQA.csv",
            ]

            dfs = []
            total_docs = 0

            for csv_file in csv_files:
                file_path = docs_path / csv_file
                if not file_path.exists():
                    logger.warning(f"CSV file not found: {file_path}")
                    continue

                df = pd.read_csv(file_path)
                logger.info(
                    f"Loaded {csv_file}: {len(df)} documents"
                )
                dfs.append(df)
                total_docs += len(df)

            if not dfs:
                logger.error("No CSV files found in docs folder")
                return False

            # Concatenate all DataFrames
            self.corpus_df = pd.concat(dfs, ignore_index=True)
            
            # Create index mapping: doc_id -> row index
            self.corpus_index = {i: i for i in range(len(self.corpus_df))}
            self.num_documents = len(self.corpus_df)

            logger.info(
                f"Corpus loaded successfully: {self.num_documents} total documents"
            )
            logger.debug(f"Corpus columns: {list(self.corpus_df.columns)}")
            
            return True

        except Exception as e:
            logger.error(f"Error loading corpus: {str(e)}")
            return False

    def get_document(self, doc_id):
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document index (integer).
            
        Returns:
            dict: Document data as dictionary, or None if not found.
        """
        try:
            if self.corpus_df is None:
                logger.warning("Corpus not loaded")
                return None

            if doc_id not in self.corpus_index:
                logger.warning(f"Document ID {doc_id} not found in corpus")
                return None

            row_idx = self.corpus_index[doc_id]
            row = self.corpus_df.iloc[row_idx]
            
            return row.to_dict()

        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None

    def get_all_documents(self):
        """
        Get all documents in corpus.
        
        Returns:
            pd.DataFrame: Corpus DataFrame or None if not loaded.
        """
        return self.corpus_df

    def search_in_corpus(self, doc_ids, limit=None):
        """
        Retrieve multiple documents by IDs.
        
        Args:
            doc_ids: List of document IDs.
            limit: Maximum number of documents to return.
            
        Returns:
            list: List of document dictionaries.
        """
        if limit is not None:
            doc_ids = doc_ids[:limit]

        documents = []
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc:
                documents.append(doc)

        return documents

    def get_document_preview(self, doc_id, max_chars=200):
        """
        Get a preview of document content.
        
        Args:
            doc_id: Document ID.
            max_chars: Maximum characters to return.
            
        Returns:
            str: Preview text or None if document not found.
        """
        doc = self.get_document(doc_id)
        if not doc:
            return None

        # Try to get preview from Answer/answer field or first available text field
        preview_text = None
        for field in ["Answer", "answer", "content", "text"]:
            if field in doc and doc[field]:
                preview_text = str(doc[field])
                break

        if preview_text is None:
            # Use first non-null field
            for key, value in doc.items():
                if value and isinstance(value, str):
                    preview_text = str(value)
                    break

        if preview_text:
            if len(preview_text) > max_chars:
                return preview_text[:max_chars] + "..."
            return preview_text

        return "No content available"

    def is_loaded(self):
        """Check if corpus is loaded."""
        return self.corpus_df is not None and len(self.corpus_df) > 0


# Global corpus instance
_corpus = None


def get_corpus():
    """Get or create global corpus instance."""
    global _corpus
    if _corpus is None:
        _corpus = CorpusLoader()
    return _corpus


def initialize_corpus(root_path=None):
    """
    Initialize the global corpus instance.
    
    Args:
        root_path: Root directory path.
        
    Returns:
        bool: True if successfully loaded.
    """
    corpus = get_corpus()
    return corpus.load_corpus(root_path)
