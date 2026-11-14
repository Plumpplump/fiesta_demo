import flet as ft

from controllers.app_controller import AppController


def main(page: ft.Page):
    page.title = "Fiesta Demo - School Uniform Check"
    page.padding = 16

    controller = AppController(page)
    controller.mount()
    page.on_disconnect = controller.on_disconnect
    page.run_task(controller.start)


if __name__ == "__main__":
    ft.run(main)