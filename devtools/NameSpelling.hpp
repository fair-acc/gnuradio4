#ifndef GNURADIO_DEVTOOLS_NAME_SPELLING_HPP
#define GNURADIO_DEVTOOLS_NAME_SPELLING_HPP

// the name forms of CLAUDE.md §3 that devtools/naming-check checks, and how a name is spelled in another form

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace devtools {

inline const std::regex kUpperCamel{"^[A-Z][A-Za-z0-9]*$"};
inline const std::regex kLowerCamel{"^[a-z][A-Za-z0-9]*$"};
inline const std::regex kSnake{"^[a-z][a-z0-9_]*$"};
inline const std::regex kUpperSnake{"^[A-Z][A-Z0-9_]*$"};
inline const std::regex kUnderscored{"^_[A-Za-z0-9_]+$"}; // `_` then either case: a non-public member, a reflected `_` field
inline const std::regex kAcronymRun{"[A-Z]{3,}"};         // three capitals in a row: an acronym written in capitals (A4)
inline const std::regex kSingleLetter{"^[A-Za-z]$"};
// std-style traits, accepted anywhere: aliases `*_t`, `is_*`/`has_*`; a member alias `*_type` as well (`value_type`)
inline const std::regex kTypeName{"^[A-Z][A-Za-z0-9]*$|^[a-z][a-z0-9_]*_t$|^(is|has)_[a-z0-9_]+$"};
inline const std::regex kMemberTypeName{"^[a-z][a-z0-9_]*_type$"};
inline const std::regex kConceptName{"^[A-Z][A-Za-z0-9]*$|^[a-z][a-z0-9_]*$"};
inline const std::regex kConstexprName{"^k[A-Z][A-Za-z0-9]*$|^[A-Za-z]$|^[a-z][a-z0-9_]*_v$|^(is|has)_[a-z0-9_]+$"};
// mathematical notation: a short capitalised symbol (`N`, `Nfft`), or a one-letter or capitalised stem with a short subscript (`x_0`, `S_k`, `sumRe_02`)
inline const std::regex kMathematical{"^[A-Z][A-Za-z0-9]{0,5}$|^([A-Za-z]|[A-Za-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)_[A-Za-z0-9]{1,4}$"};
inline const std::regex kNonTypeParameter{"^[a-z][A-Za-z0-9]*$|^[A-Z][A-Za-z0-9]*$|^[A-Z][A-Z0-9_]*$"};
inline const std::regex kDirectoryName{"^[a-z0-9][a-z0-9_.-]*$"};

/// a name's words, lowercase: split at `_` and where the case changes (`parseHTTPHeader` is parse, http, header)
inline std::vector<std::string> wordsOf(std::string_view name) {
    std::vector<std::string> words;
    std::string              word;
    for (std::size_t i = 0; i < name.size(); ++i) {
        const auto character  = static_cast<unsigned char>(name[i]);
        const bool startsWord = character == '_' || ((std::isupper(character) != 0) && i > 0 && ((std::islower(static_cast<unsigned char>(name[i - 1])) != 0) || (i + 1 < name.size() && (std::islower(static_cast<unsigned char>(name[i + 1])) != 0))));
        if (startsWord && !word.empty()) {
            words.push_back(word);
            word.clear();
        }
        if (character != '_') {
            word.push_back(static_cast<char>(std::tolower(character)));
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    return words;
}

using Words = std::vector<std::string>;

enum class Capitals { afterFirst, all };

/// the words run together, each capitalised after the first (`itemCount`), or every one (`ItemCount`)
inline std::string camelCased(const Words& words, Capitals capitals) {
    std::string camel;
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::string word = words[i];
        if ((i > 0 || capitals == Capitals::all) && !word.empty()) {
            word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
        }
        camel += word;
    }
    return camel;
}

inline std::string joinedWith(const Words& words, std::string_view separator) {
    std::string joinedName;
    for (std::size_t i = 0; i < words.size(); ++i) {
        joinedName += (i > 0 ? std::string{separator} : std::string{}) + words[i];
    }
    return joinedName;
}

/// a form a name can be proposed in: its name in a report, the pattern a name in it matches, and how words are spelled in it
struct NameForm {
    std::string_view  name;
    const std::regex& pattern;
    std::string (*spell)(const Words& words);
};

inline const std::array<NameForm, 6>& nameForms() {
    static const std::regex              kUnderscoredLowerCamel{"^_[a-z][A-Za-z0-9]*$"};
    static const std::regex              kConstant{"^k[A-Z][A-Za-z0-9]*$"};
    static const std::array<NameForm, 6> kForms{{
        {"UpperCamel", kUpperCamel, [](const Words& words) { return camelCased(words, Capitals::all); }},
        {"lowerCamel", kLowerCamel, [](const Words& words) { return camelCased(words, Capitals::afterFirst); }},
        {"_lowerCamel", kUnderscoredLowerCamel, [](const Words& words) { return "_" + camelCased(words, Capitals::afterFirst); }},
        {"snake", kSnake, [](const Words& words) { return joinedWith(words, "_"); }},
        {"kConstant", kConstant, [](const Words& words) { return "k" + camelCased(words, Capitals::all); }},
        {"UPPER_SNAKE", kUpperSnake,
            [](const Words& words) {
                std::string upperSnake = joinedWith(words, "_");
                std::ranges::transform(upperSnake, upperSnake.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                return upperSnake;
            }},
    }};
    return kForms;
}

/// empty for an unknown form
inline std::string spellingFor(std::string_view name, std::string_view form) {
    const auto found = std::ranges::find(nameForms(), form, &NameForm::name);
    if (found == nameForms().end()) {
        return {};
    }
    const std::string        withoutLeadingUnderscores = std::string{name}.erase(0, name.find_first_not_of('_'));
    std::vector<std::string> words                     = wordsOf(withoutLeadingUnderscores);
    if (form == "kConstant" && words.size() > 1 && words.front() == "k") {
        words.erase(words.begin());
    }
    return found->spell(words);
}

/// for an unknown form, a pattern nothing matches
inline const std::regex& patternOf(std::string_view form) {
    static const std::regex kNever{"$^"};
    const auto              found = std::ranges::find(nameForms(), form, &NameForm::name);
    return found == nameForms().end() ? kNever : found->pattern;
}

} // namespace devtools

#endif // GNURADIO_DEVTOOLS_NAME_SPELLING_HPP
