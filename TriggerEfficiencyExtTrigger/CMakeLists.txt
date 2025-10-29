project(alpaca_module_TriggerEfficiencyExtTrigger)

set(MODULE_NAME TriggerEfficiencyExtTrigger)



file(GLOB CURRENT_MODULE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
file(GLOB CURRENT_MODULE_INC "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

list(APPEND CURRENT_MODULE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME}Main.cxx")

alpaca_configure_module(${MODULE_NAME})
alpaca_build_module_with_RQs(${MODULE_NAME})
#alpaca_build_module_with_SimsRQs(${MODULE_NAME})


alpaca_add_executable(${MODULE_NAME} ${CURRENT_MODULE_SRC} ${CURRENT_MODULE_INC} )

alpaca_target_link_libraries(${MODULE_NAME} PUBLIC
    ${ROOT_TARGETS}
    
    )

alpaca_target_include_directories(${MODULE_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

