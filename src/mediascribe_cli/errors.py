class CLIError(Exception):
    """Expected user-facing CLI failure with a stable process exit code."""

    exit_code = 2


class UsageError(CLIError):
    exit_code = 2


class ConfigError(CLIError):
    """Invalid or unsafe configuration (env files, provider endpoints)."""

    exit_code = 2

class DependencyError(CLIError):
    exit_code = 3


class ProviderError(CLIError):
    exit_code = 4


class PartialFailure(CLIError):
    exit_code = 5
