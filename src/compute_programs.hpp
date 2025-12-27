#pragma once
#include <unordered_map>
#include <string>
#include <glad/gl.h>

class ComputePrograms
{
public:
    static ComputePrograms& instance();

    void load(const std::string& name, const std::string& path);
    void loadManifest(const std::string& path);
    GLuint get(const std::string& name) const;

private:
    ComputePrograms() = default;
    ~ComputePrograms();

    static void checkShader(GLuint shader, const std::string& name);
    static void checkProgram(GLuint program, const std::string& name);

    std::unordered_map<std::string, GLuint> programs;
};