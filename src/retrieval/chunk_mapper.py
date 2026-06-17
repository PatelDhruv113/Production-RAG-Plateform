import json

class ChunkMapper:

  def save_mapping(self, mapping, path):
  
      with open(path, "w") as f:
  
          json.dump(mapping, f)
  
  def load_mapping(self, path):
  
      with open(path, "r") as f:
  
          return json.load(f)