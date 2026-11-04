from meeting_notes.bootstrap import setup_runtime_environment

setup_runtime_environment()

from meeting_notes.ui import MeetingNotesApp


def main() -> None:
    app = MeetingNotesApp()
    app.launch()


if __name__ == "__main__":
    main()
