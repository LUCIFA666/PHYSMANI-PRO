"""Validate the original TaskSpec optional-after-native trailing phase contract."""


def require_phase_prefix(skills,declared,omitted):
    declared=tuple(declared);full=tuple(s.name for s in skills)
    tail=tuple(skills[len(declared):])
    if (not declared or declared!=full[:len(declared)] or len(declared)>len(full)
            or any(not s.optional_after_native_completion for s in tail)
            or [row['phase'] for row in omitted]!=[s.name for s in tail]):
        raise ValueError('all mandatory phases and only original native-optional trailing omissions required')
    return declared
