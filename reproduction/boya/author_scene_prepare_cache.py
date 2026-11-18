"""Preparation-only exact file/mesh reuse; no physics or sampling changes."""
from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch

import author_collection_scenes as S


class PreparationCache:
    def __init__(self):
        self.files={};self.geometry={};self.hashes=0;self.stat_checks=0

    @staticmethod
    def signature(path):
        s=Path(path).stat()
        return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

    def verify(self,bindings):
        for path,expected in bindings.items():
            key=str(Path(path).resolve());before=self.signature(key);self.stat_checks+=1
            cached=self.files.get(key)
            # Filesystem timestamps may have coarse update granularity. Always
            # hash recently written files, including same-size rapid rewrites.
            stable_age=time.time_ns()-max(before[-2:])>=1_000_000_000
            if cached is not None and cached[0]==before and stable_age:
                actual=cached[1]
            else:
                actual=S.sha(key);self.hashes+=1
                if self.signature(key)!=before:raise ValueError('Dependency changed during preparation hash: '+key)
                self.files[key]=(before,actual)
            if actual!=expected:raise ValueError('Frozen scene dependency changed: '+key)

    @contextmanager
    def installed(self):
        import author_scene
        original=S.sample_scene
        def sample(*args,**kwargs):
            if kwargs.get('geometry_cache') is None:kwargs['geometry_cache']=self.geometry
            return original(*args,**kwargs)
        with patch.object(S,'verify',self.verify),patch.object(author_scene,'verify',self.verify),patch.object(S,'sample_scene',sample):
            yield self

    def receipt(self):
        return dict(file_hashes=self.hashes,file_stat_checks=self.stat_checks,cached_object_geometries=len(self.geometry),
            sampling_algorithm_changed=False,source_vertices_changed=False,physics_steps=0,
            final_uncached_verification_required=True)
