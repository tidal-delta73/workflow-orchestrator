"""Deterministic generation of permuted definition documents.

Used to prove that planning and diagnostic selection depend only on the
*content* of a definition, never on the order tasks or dependencies are
listed in.
"""
import itertools
import json


def unique_permutations(items):
    """All distinct permutations of a list, collapsing equal items.

    Order of the returned list is deterministic (it follows
    ``itertools.permutations`` on the input order).
    """
    seen = set()
    result = []
    for perm in itertools.permutations(items):
        key = json.dumps(list(perm), ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(list(perm))
    return result


def _task(entry):
    task_id, deps = entry
    task = {"id": task_id}
    if deps:
        task["depends_on"] = list(deps)
    return task


def equivalent_documents(tasks, *, duplicate_deps=False):
    """Every task-ordering x depends_on-ordering of a valid task set.

    ``tasks`` is a list of ``{"id", "depends_on"?}`` dicts with unique ids
    and already de-duplicated dependency lists. When ``duplicate_deps`` is
    true, one extra variant per task ordering is added with a dependency
    item repeated (non-adjacently where possible); duplicates are
    semantically irrelevant and must not change the plan.
    """
    base_deps = {
        task["id"]: list(task.get("depends_on", [])) for task in tasks
    }

    documents = []
    for order in unique_permutations(list(base_deps)):
        options = []
        for task_id in order:
            dep_orders = unique_permutations(base_deps[task_id]) or [[]]
            options.append(
                [(task_id, dep_order) for dep_order in dep_orders]
            )
        for combo in itertools.product(*options):
            documents.append({"tasks": [_task(entry) for entry in combo]})

    if duplicate_deps:
        for order in unique_permutations(list(base_deps)):
            entries = []
            for task_id in order:
                deps = base_deps[task_id]
                if deps:
                    # Repeat the first item at the end: adjacent for a
                    # single dependency, non-adjacent for several.
                    deps = [*deps, deps[0]]
                entries.append((task_id, deps))
            documents.append({"tasks": [_task(entry) for entry in entries]})

    return documents


def spread(items, count):
    """Deterministically choose ``count`` positions spread across ``items``.

    Always includes the first and last elements. Used to exercise "several
    input permutations" for large entry sets without enumerating hundreds of
    subprocess runs; hash-seed parametrization covers the remaining
    order-sensitivity surface.
    """
    if len(items) <= count:
        return list(items)
    picked = [items[int(i * (len(items) - 1) / (count - 1))] for i in range(count)]
    return picked
