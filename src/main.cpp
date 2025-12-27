#include <memory.h>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>

#include <iostream>

constexpr unsigned WINDOW_WIDTH  = 1280;
constexpr unsigned WINDOW_HEIGHT = 720;

sf::Window createWindow(bool fullscreen)
{
    sf::ContextSettings settings;
    settings.majorVersion   = 4;
    settings.minorVersion   = 3;
    settings.attributeFlags = sf::ContextSettings::Core;

    sf::VideoMode mode = fullscreen
        ? sf::VideoMode::getDesktopMode()
        : sf::VideoMode({ WINDOW_WIDTH, WINDOW_HEIGHT });

    auto style = fullscreen
        ? sf::Style::None
        : sf::Style::Titlebar | sf::Style::Close;

    auto state = fullscreen
        ? sf::State::Fullscreen
        : sf::State::Windowed;

    sf::Window window(mode, "Ouroboros", style, state, settings);
    window.setVerticalSyncEnabled(false);

    return window;
}

int main()
{
    bool fullscreen = false;
    sf::Window window = createWindow(fullscreen);

    bool running = true;

    while (running)
    {
        while (auto event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                running = false;
            }
            else if (const auto* key =
                         event->getIf<sf::Event::KeyPressed>())
            {
                if (key->scancode == sf::Keyboard::Scancode::Escape)
                {
                    running = false;
                }
                else if (key->scancode == sf::Keyboard::Scancode::F11)
                {
                    fullscreen = !fullscreen;
                    window.close();
                    window = createWindow(fullscreen);
                }
            }
        }

        // Compute-only for now
        window.display();
    }

    return 0;
}
