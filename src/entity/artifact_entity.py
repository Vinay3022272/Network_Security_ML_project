from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str
    
    # this is same as the upper one
# class DataIngestionArtifact:
#     def __init__(self, trained_file_path: str, test_file_path: str):
#         self.trained_file_path = trained_file_path
#         self.test_file_path = test_file_path
    