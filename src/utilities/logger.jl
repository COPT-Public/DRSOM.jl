module Logger
"""
A module for initializing a logging system.

This module creates multiple loggers that write to different files and formats the log messages in a specific way.
The log files are saved in a folder specified by the constant `LOG_FOLDER`.

Reference: https://juliacheat.codes/tutorials/getting-started-logging-julia/
"""

# Import required packages
using Logging
using LoggingExtras
using LoggingFormats
using Dates

# Define constants
const DEFAULT_LOGGER = current_logger() # Refers to the current logger
const DATE_FORMAT = dateformat"yyyy-mm-ddTHH:MM:SS" # Specifies the format to use for dates in log messages
const PARENT_MODULE = parentmodule(@__MODULE__) # Refers to the parent module of the current module
const LOG_FOLDER =
  isnothing(pkgdir(PARENT_MODULE)) ? joinpath(@__DIR__, "log") : joinpath(pkgdir(PARENT_MODULE), "log/") # Specifies the folder where log files will be saved

"""
A function to filter log messages by module.

This function takes a log object and returns `true` if the module of the log is equal to the parent module of the current module.

Args:

  - log (LogRecord): The log message to be filtered.

Returns:

  - bool: `true` if the module of the log is equal to the current module or parent module of the current module, `false` otherwise.
"""
function module_message_filter(log)
  # println(log._module)
  # log._module !== nothing && (log._module === PARENT_MODULE || parentmodule(log._module) === PARENT_MODULE)
  return true
end

"""
A function to create a file logger.

This function creates a logger that logs messages to a file with the specified name in the `LOG_FOLDER` directory.
The logger formats the log messages in a specific way and includes the current date, log level, filename, line number, and message.

Kwargs:

  - name (str): The name of the log file. Defaults to "info".
  - exceptions (bool): Whether or not to print the exception in the log message. Defaults to `true`.

Returns:

  - FormatLogger: A logger that logs messages to a file with the specified name in the `LOG_FOLDER` directory.
"""
function file_logger(; name="info", exceptions=true)
  # The FormatLogger constructor takes a file path and a function that formats log messages
  FormatLogger(joinpath(LOG_FOLDER, "$name.log"); append=false) do io, args
    # Use datetime in log messages in files
    date = Dates.format(now(), DATE_FORMAT)
    # pad level, filename and lineno so things look nice
    level = rpad(args.level, 5, " ")
    filename = lpad(basename(args.file), 10, " ")
    lineno = rpad(args.line, 5, " ")
    message = args.message
    # Write the formatted log message to the file
    println(io, "$date | $level | $filename:$lineno - $message")
    # If the log message includes an exception, print it explicitly
    if exceptions && :exception ∈ keys(args.kwargs)
      e, stacktrace = args.kwargs[:exception]
      println(io, "exception = ")
      showerror(io, e, stacktrace)
      println(io)
    end
  end
end

"""
A function to add the filename to log messages.

This function takes a logger and returns a new logger that includes the filename of the log message in the log message.

Args:

  - logger (AbstractLogger): The logger to transform.

Returns:

  - TransformerLogger: A new logger that includes the filename of the log message in the log message.
"""
function filename_logger(logger)
  TransformerLogger(logger) do log
    merge(log, (; message="$(basename(log.file)) - $(log.message)"))
  end
end

"""
A function to initialize the logger.

This function initializes the logging system by creating the log folder if it doesn't already exist and setting up multiple loggers that log messages with different levels to different log files.
It also logs a message to indicate that the logger has been initialized.
"""
function initialize()
  # Create the log folder if it doesn't already exist
  isdir(LOG_FOLDER) || mkpath(LOG_FOLDER)
  # Initialize the global logger with several loggers:
  global_logger(
    # A logger that logs messages from the current module with a minimum level of Info to a file called "info.log" in the LOG_FOLDER directory
    TeeLogger(
      EarlyFilteredLogger(
        module_message_filter,
        MinLevelLogger(file_logger(; name="info", exceptions=false), Logging.Info),
      ),
      # A logger that logs messages with a minimum level of Debug to a file called "debug.log" in the LOG_FOLDER directory
      EarlyFilteredLogger(module_message_filter, MinLevelLogger(file_logger(; name="debug"), Logging.Debug)),
      # A logger that logs messages from the current module with the filename appended to the message to the current logger
      EarlyFilteredLogger(module_message_filter, filename_logger(DEFAULT_LOGGER)),
      # A logger that logs messages from other modules to the current logger
      EarlyFilteredLogger(module_message_filter, DEFAULT_LOGGER),
    ),
  )
  # Log a message to indicate that the logger has been initialized
  @info "Initialized logger"
  nothing
end

"""
A function to reset the logger.

This function resets the global logger to the original logger.
"""
function reset()
  # Reset the global logger to the original logger
  global_logger(DEFAULT_LOGGER)
  nothing
end

end # module Logger