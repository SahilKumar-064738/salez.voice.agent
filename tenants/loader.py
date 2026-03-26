import json
import os
from abc import ABC, abstractmethod


# -------- BASE INTERFACE --------
class TenantStore(ABC):
    @abstractmethod
    def get(self, name: str) -> dict | None:
        pass

    @abstractmethod
    def reload(self):
        pass


# -------- FILE-BASED STORE --------
class FileTenantStore(TenantStore):
    def __init__(self, base_path="tenants"):
        self.base_path = base_path
        self._data: dict = {}
        self.reload()

    def reload(self):
        new_data = {}

        for file in os.listdir(self.base_path):
            if not file.endswith(".json"):
                continue

            name = file.replace(".json", "")

            try:
                with open(os.path.join(self.base_path, file)) as f:
                    d = json.load(f)

                # Preprocess FAQ for fast lookup
                d["_faq_index"] = [
                    {
                        "q_words": set(faq["question"].lower().split()),
                        "answer": faq["answer"]
                    }
                    for faq in d.get("faqs", [])
                ]

                new_data[name] = d

            except Exception as e:
                print(f"❌ Tenant load failed [{name}]: {e}")

        # atomic swap (VERY IMPORTANT)
        self._data = new_data
        print(f"✅ Loaded {len(self._data)} tenants")

    def get(self, name: str) -> dict | None:
        return self._data.get(name)


# -------- SINGLETON --------
_store: TenantStore | None = None


def load_tenants():
    global _store
    _store = FileTenantStore()


def get_tenant(name: str) -> dict | None:
    if _store is None:
        return None
    return _store.get(name)


def reload_tenants():
    global _store

    if _store is None:
        _store = FileTenantStore()
    else:
        _store.reload()